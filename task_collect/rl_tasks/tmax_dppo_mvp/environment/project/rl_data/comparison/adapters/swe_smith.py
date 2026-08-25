"""Adapter for ``hamishivi/agent-task-swe-smith`` (SWE-smith, ~59k tasks).

SWE-smith (``SWE-bench/SWE-smith``, arXiv:2504.21798) is a SWE-bench-style
bug-fixing benchmark generated at scale by *synthetic bug injection*: for each
of ~250 real Python repos a single "environment" Docker image bakes the repo at
a base commit into ``/testbed`` (inside a conda env named ``testbed``), with a
passing unit-test suite. Thousands of synthetic bugs are then generated per repo
by procedurally corrupting the *source code* (combine-file/func rewrites, etc.).
Each published task is the *repair* problem: the agent is dropped into the buggy
repo and must patch ``/testbed`` so a set of broken unit tests (``FAIL_TO_PASS``)
passes again.

Two HF sources, joined on ``task_id`` == ``instance_id``
--------------------------------------------------------
The dataset the comparison consumes (``hamishivi/agent-task-swe-smith``) is the
open-instruct RLVR packaging: each parquet row carries the *shared base* Docker
image and a chat-formatted prompt, but **not** the verifier nor the bug:

    messages     [system, user]   # user msg embeds the vanillux workflow boilerplate
    ground_truth <task_id slug>
    dataset      "passthrough"     # reward is env-computed (PassthroughVerifier)
    env_config   {env_name: "swerl_vanillux_sandbox",
                  image:    "jyangballin/swesmith.x86_64.<repo>.<sha>",
                  task_id:  "<slug>"}
    source       "swe_smith"

The authoritative verifier + bug live in the upstream release
(``SWE-bench/SWE-smith``, ~52k rows), keyed by the same slug as ``instance_id``:

    instance_id   <slug>
    patch         # the diff that, when applied, *creates the bug*
    FAIL_TO_PASS  # unit tests the bug breaks (the repair target)
    PASS_TO_PASS  # unit tests that must keep passing (no-regression set)
    image_name    # jyangballin/swesmith.x86_64.<repo>.<sha>  (== env_config.image)
    repo          # swesmith/<repo>.<sha>

The base image is **shared** across every bug instance of a given ``<repo>.<sha>``
(one image, thousands of tasks), so unlike CLI-Gym (which ships a pre-corrupted
per-task image) the per-instance bug is *not* baked in — we apply it ourselves at
build time (see "Bug injection" below).

The task **instruction** is recovered from the hamishivi user message (the
SWE-smith problem statement), stripped of the vanillux workflow boilerplate our
own harness re-adds — byte-identical to the CLI-Gym handling.

Bug injection (container.def ``%post``)
---------------------------------------
The SWE-smith dataset ``patch`` is, by construction, the diff *that creates the
bug* (clean ``/testbed`` -> buggy). The base image ships the clean repo, so we
``git apply`` that patch during ``apptainer build`` to materialize the buggy
state the agent must repair. The build *fails loudly* (no ``container.sif``) if
the patch does not apply, so the prebuild/solve harness simply skips an
un-injectable task rather than silently shipping a trivially-passing one. The
project is installed editable in the ``testbed`` env, so the source edit is live
without a reinstall (validated end-to-end during ingest smoke tests).

Verifier contract (test_final_state.py)
----------------------------------------
SWE-bench resolution is "all ``FAIL_TO_PASS`` pass (the bug is fixed) **and** all
``PASS_TO_PASS`` still pass (no regressions)". We mirror CLI-Gym's selected-test
verifier and run ``FAIL_TO_PASS`` by default (the direct "did you fix the bug"
signal; ``PASS_TO_PASS`` is ~500-670 tests/task and dominates runtime). The
``PASS_TO_PASS`` set is baked into the wrapper too and is additionally checked
when ``SWE_SMITH_CHECK_P2P`` is set in the container env
(``APPTAINERENV_SWE_SMITH_CHECK_P2P=1``). The wrapper shells out to::

    source /opt/miniconda3/bin/activate; conda activate testbed; cd /testbed
    pytest --disable-warnings --color=no --tb=short -p no:cacheprovider <targets>

and asserts ``returncode == 0`` (pytest exits non-zero iff any target fails or
errors).

The agent's fixes persist because the solve harness keeps a single
``--writable-tmpfs`` Apptainer instance alive across the rollout *and* the final
test (the editable ``/testbed`` checkout and the conda env are writable in the
overlay and visible to the verifier).

Wrinkles vs. the other adapters
-------------------------------
1.  **Two-source join, not a tarball/snapshot** (like CLI-Gym): two parquet
    datasets read via ``datasets.load_dataset`` and joined in memory on the slug.

2.  **Shared base image + build-time bug injection.** Many tasks ``FROM`` the
    same ``jyangballin/swesmith.x86_64.<repo>.<sha>`` image (public on Docker
    Hub), so the per-task SIFs differ only by the injected bug layer. The solve
    script's prebuild pulls each (with apptainer's layer cache amortizing the
    shared base across same-repo tasks).

3.  **Outer-harness pytest on the base-conda PATH** (identical to CLI-Gym): the
    harness runs ``pytest pytest_final_state.py`` from the base-conda PATH, which
    ships no pytest; the container.def ``%post`` makes one reachable purely so the
    wrapper can be collected (it re-activates ``testbed`` for the real run).

CLI:

    python -m rl_data.comparison.adapters.swe_smith \\
        --dst rl_data/output/tasks_swe_smith

Options mirror the other adapters (``--limit``, ``--workers``,
``--skip-download``, ``--revision``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rl_data.comparison.adapters import (
    Adapter,
    _PLACEHOLDER_INITIAL_STATE,
    register_adapter,
)

# Reuse CLI-Gym's instruction extraction + base-conda pytest bootstrap: the
# hamishivi packaging (lead-in, boilerplate marker) and the SWE-Smith base
# images (/testbed in a `testbed` conda env, no base-conda pytest) are identical.
from rl_data.comparison.adapters.cli_gym import (
    _PYTEST_BASE_BOOTSTRAP_POST,
    _instruction_from_messages,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Shared base images + open-instruct RLVR prompts (the dataset we integrate).
HF_REPO_ID = "hamishivi/agent-task-swe-smith"
# Upstream release carrying the authoritative bug patch + FAIL/PASS_TO_PASS,
# keyed by the same slug (``instance_id`` == hamishivi ``task_id``).
VERIFIER_REPO_ID = "SWE-bench/SWE-smith"


# ---------------------------------------------------------------------------
# container.def %post: apply the SWE-smith bug patch (clean -> buggy /testbed),
# then make a pytest reachable on the base-conda PATH for the outer wrapper.
#
# The bug-apply is strict (set -e): if the patch does not apply, the build
# aborts -> no container.sif -> the harness skips this task, which is exactly
# what we want (better to drop an un-injectable task than to ship one whose
# FAIL_TO_PASS tests already pass). We try git apply, then a 3-way merge, then
# GNU patch with fuzz, before giving up.
# ---------------------------------------------------------------------------
_BUG_APPLY_POST = r"""    set -e
    # --- SWE-smith bug injection ------------------------------------------
    # The dataset `patch` is the diff that CREATES the bug (clean -> buggy).
    # The base image ships the clean repo at /testbed; apply the bug here so
    # the agent has something to repair and FAIL_TO_PASS starts red.
    if [ ! -d /testbed ]; then
        echo "ERROR: /testbed missing in base image; cannot inject SWE-smith bug" >&2
        exit 1
    fi
    cd /testbed
    _bug=/opt/swesmith_bug.patch
    if git -C /testbed rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git apply -v "$_bug" 2>/tmp/swesmith_bugapply.log \
            || git apply -v --3way "$_bug" 2>>/tmp/swesmith_bugapply.log \
            || patch -p1 --fuzz=3 < "$_bug" 2>>/tmp/swesmith_bugapply.log \
            || { echo "ERROR: SWE-smith bug patch did not apply:" >&2; \
                 cat /tmp/swesmith_bugapply.log >&2; exit 1; }
    else
        patch -p1 --fuzz=3 < "$_bug" 2>/tmp/swesmith_bugapply.log \
            || { echo "ERROR: SWE-smith bug patch did not apply (no git):" >&2; \
                 cat /tmp/swesmith_bugapply.log >&2; exit 1; }
    fi
"""


# ---------------------------------------------------------------------------
# Pytest verifier wrapper (test_final_state.py).
#
# Shells out to the testbed-env pytest over FAIL_TO_PASS (the bug's broken
# tests) and asserts they all pass. PASS_TO_PASS is baked in too and added to
# the run when SWE_SMITH_CHECK_P2P is set in the container env. The JSON lists
# are substituted at ingest.
# ---------------------------------------------------------------------------
_TEST_FINAL_WRAPPER = r'''"""Pytest wrapper around SWE-smith's gold unit-test verifier.

SWE-smith tasks are real Python repos (installed editable at ``/testbed`` in the
``testbed`` conda env) with a single synthetic bug injected at build time. The
task is resolved iff, after the agent's repair, the broken unit tests
(``FAIL_TO_PASS``) pass again (and, optionally, the ``PASS_TO_PASS`` no-regression
set still passes).

We replicate the upstream eval command exactly -- activate the ``testbed`` conda
env and run ``pytest`` over the target node ids -- and surface the outcome to our
pytest-based harness via the process exit code (pytest exits non-zero iff any
target test fails or errors).
"""
import json
import os
import shlex
import subprocess

# Baked at ingest from the SWE-bench/SWE-smith verifier row.
FAIL_TO_PASS = json.loads(r"""{F2P_JSON}""")
PASS_TO_PASS = json.loads(r"""{P2P_JSON}""")

_ACTIVATE = "source /opt/miniconda3/bin/activate; conda activate testbed"
_PYTEST_FLAGS = "--disable-warnings --color=no --tb=short -p no:cacheprovider"


def _run(targets):
    quoted = " ".join(shlex.quote(t) for t in targets)
    cmd = f"{_ACTIVATE}; cd /testbed; pytest {_PYTEST_FLAGS} {quoted}"
    return subprocess.run(["bash", "-c", cmd], check=False,
                          capture_output=True, text=True)


def _p2p_enabled():
    return os.environ.get("SWE_SMITH_CHECK_P2P", "").lower() in ("1", "true", "yes")


def test_swe_smith_verifier():
    assert FAIL_TO_PASS, "no FAIL_TO_PASS tests baked into this task"
    targets = list(FAIL_TO_PASS)
    if _p2p_enabled():
        targets += PASS_TO_PASS
    proc = _run(targets)
    assert proc.returncode == 0, (
        "SWE-smith verifier: not all target unit tests passed "
        f"(pytest exit={proc.returncode}, {len(FAIL_TO_PASS)} FAIL_TO_PASS"
        f"{' + ' + str(len(PASS_TO_PASS)) + ' PASS_TO_PASS' if _p2p_enabled() else ''}"
        " target(s)).\n"
        "stdout tail:\n" + (proc.stdout or "")[-3000:] + "\n"
        "stderr tail:\n" + (proc.stderr or "")[-1000:]
    )
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ``oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc`` ->
# repo token ``oauthlib__oauthlib.1fd52536`` + bug-strategy ``combine_file``.
_SLUG_RE = re.compile(r"^(?P<repo>.+?\.[0-9a-f]+)\.(?P<strategy>[a-zA-Z0-9_]+?)__[0-9a-z]+$")


def _parse_slug(slug: str) -> Tuple[str, str]:
    m = _SLUG_RE.match(slug)
    if not m:
        return slug, ""
    return m.group("repo"), m.group("strategy")


def _as_list(v: Any) -> List[str]:
    """Normalize a FAIL/PASS_TO_PASS cell into a list[str].

    The HF column is a list; tolerate a JSON-encoded string just in case.
    """
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return [v] if v else []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x)]
    return []


def _build_container_def(*, base_image: str, bug_patch_abs: Path) -> str:
    """Apptainer container.def for one SWE-smith task.

    Bootstraps from the shared base image, ships the bug patch via ``%files``,
    applies it in ``%post`` (clean -> buggy /testbed), then layers the
    outer-harness pytest bootstrap.
    """
    lines: List[str] = [
        "Bootstrap: docker",
        f"From: {base_image}",
        "",
        "%files",
        f"    {bug_patch_abs} /opt/swesmith_bug.patch",
        "",
        "%post",
        _BUG_APPLY_POST.rstrip(),
        # _PYTEST_BASE_BOOTSTRAP_POST opens with `set +e` and closes with the
        # mkdir/chmod /home/user convention generate_solutions expects.
        _PYTEST_BASE_BOOTSTRAP_POST.rstrip(),
        "",
        "%labels",
        "    Author swe-smith-adapter",
        f"    BaseImage {base_image}",
        '    Description "SWE-smith synthetic bug-repair task (build-time bug injection)"',
        "",
    ]
    return "\n".join(lines) + "\n"


def _sanitize_name(task_id: str) -> str:
    """``swesmith_`` + slug, with whitespace/path separators normalized."""
    slug = re.sub(r"\s+", "_", task_id.strip())
    slug = slug.replace("/", "_")
    return "swesmith_" + slug


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SweSmithAdapter(Adapter):
    """``hamishivi/agent-task-swe-smith`` (shared base images + synthetic bugs).

    Two-source adapter: images/prompts from ``hamishivi/agent-task-swe-smith``,
    bug patches + FAIL/PASS_TO_PASS from ``SWE-bench/SWE-smith``, joined on the
    instance slug.
    """

    name = "swe_smith"
    hf_repo_id = HF_REPO_ID
    default_dst = "rl_data/output/tasks_swe_smith"

    # The base-class dir-walking convert pipeline does not apply (the source is
    # two parquet tables, not task directories). Override with a row-based flow.
    def convert_one(self, src: Path, dst_root: Path) -> Optional[str]:  # pragma: no cover
        raise NotImplementedError("SweSmithAdapter converts joined rows, not dirs")

    # -- Build the (small) hamishivi index ---------------------------------
    def build_ham_index(
        self,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Index ``hamishivi/agent-task-swe-smith`` (the *small*, ~84 MB side)
        by ``task_id`` -> {image, instruction}.

        We index this side rather than the verifier side because it is two
        orders of magnitude smaller; the 4 GB verifier dataset is then read
        shard-by-shard via pyarrow (see :meth:`convert_streaming`) and never
        materialized as a Python dict, which would otherwise blow up memory
        (``PASS_TO_PASS`` alone is ~500-670 strings per row over ~52k rows).

        We deliberately bypass ``datasets.load_dataset`` here: in non-streaming
        mode it builds a multi-GB Arrow cache (OOM-prone on a login node), and
        in streaming mode it imports ``torch`` (which fails to load CUDA libs on
        a GPU-less login node). Reading the published parquet directly with
        pyarrow avoids both.
        """
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files

        hf_cache = str((cache_dir / "_hf")) if cache_dir else None
        files = sorted(
            f for f in list_repo_files(self.hf_repo_id, repo_type="dataset",
                                       revision=revision)
            if f.endswith(".parquet")
        )
        logger.info("Indexing %s from %d parquet shard(s) ...",
                    self.hf_repo_id, len(files))
        index: Dict[str, Dict[str, str]] = {}
        for fn in files:
            p = hf_hub_download(repo_id=self.hf_repo_id, repo_type="dataset",
                                filename=fn, revision=revision, cache_dir=hf_cache)
            pf = pq.ParquetFile(p)
            for arrow_batch in pf.iter_batches(
                batch_size=4000, columns=["messages", "env_config", "ground_truth"]
            ):
                for r in arrow_batch.to_pylist():
                    env_cfg = r.get("env_config") or {}
                    task_id = env_cfg.get("task_id") or r.get("ground_truth")
                    image = env_cfg.get("image")
                    if not task_id or not image:
                        continue
                    index[task_id] = {
                        "image": image,
                        "instruction": _instruction_from_messages(r.get("messages")),
                    }
        logger.info("Indexed %d hamishivi task(s).", len(index))
        return index

    # -- Stream the verifier parquet shards + convert joined rows ----------
    def convert_streaming(
        self,
        dst_root: Path,
        *,
        limit: int = 0,
        workers: int = 16,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        batch_size: int = 2000,
    ) -> Tuple[int, int]:
        """Join + convert without holding the 4 GB verifier dataset in RAM.

        Reads ``SWE-bench/SWE-smith`` shard-by-shard via pyarrow (one parquet at
        a time, ``iter_batches`` bounds the live ``PASS_TO_PASS`` footprint),
        joins each row against the small hamishivi index, and converts each
        arrow batch immediately. With ``--limit`` we stop after the first shard
        that satisfies the cap, so a smoke test only downloads the first shard.
        Tasks absent from the hamishivi packaging, or with an empty bug patch /
        empty FAIL_TO_PASS, are dropped with a count.
        """
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files

        dst_root.mkdir(parents=True, exist_ok=True)
        hf_cache = str((cache_dir / "_hf")) if cache_dir else None

        ham_index = self.build_ham_index(revision=revision, cache_dir=cache_dir)

        files = sorted(
            f for f in list_repo_files(VERIFIER_REPO_ID, repo_type="dataset")
            if f.endswith(".parquet")
        )
        logger.info("Joining against %s (%d parquet shard(s)) ...",
                    VERIFIER_REPO_ID, len(files))

        converted = 0
        skipped = 0
        n_not_in_ham = 0
        n_empty_patch = 0
        n_empty_f2p = 0
        submitted = 0

        def _flush(rows: List[Dict[str, Any]]) -> None:
            nonlocal converted, skipped
            if not rows:
                return
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(self.convert_row, row, dst_root): row
                        for row in rows}
                for fut in as_completed(futs):
                    try:
                        r = fut.result()
                    except Exception as e:
                        logger.warning("convert_row failed on %s: %s",
                                       futs[fut].get("task_id"), e)
                        r = None
                    if r is not None:
                        converted += 1
                    else:
                        skipped += 1

        _cols = ["instance_id", "patch", "FAIL_TO_PASS", "PASS_TO_PASS",
                 "image_name", "repo"]
        stop = False
        for fn in files:
            if stop:
                break
            logger.info("Reading verifier shard %s ...", fn)
            p = hf_hub_download(repo_id=VERIFIER_REPO_ID, repo_type="dataset",
                                filename=fn, cache_dir=hf_cache)
            pf = pq.ParquetFile(p)
            for arrow_batch in pf.iter_batches(batch_size=batch_size, columns=_cols):
                batch: List[Dict[str, Any]] = []
                for r in arrow_batch.to_pylist():
                    iid = r.get("instance_id")
                    h = ham_index.get(iid) if iid else None
                    if h is None:
                        n_not_in_ham += 1
                        continue
                    bug_patch = r.get("patch") or ""
                    if not bug_patch.strip():
                        n_empty_patch += 1
                        continue
                    f2p = _as_list(r.get("FAIL_TO_PASS"))
                    if not f2p:
                        n_empty_f2p += 1
                        continue
                    repo_token, strategy = _parse_slug(iid)
                    batch.append({
                        "task_id": iid,
                        # Prefer the hamishivi env image; fall back to the
                        # verifier's image_name (identical in practice).
                        "image": h.get("image") or (r.get("image_name") or ""),
                        "instruction": h.get("instruction", ""),
                        "bug_patch": bug_patch,
                        "fail_to_pass": f2p,
                        "pass_to_pass": _as_list(r.get("PASS_TO_PASS")),
                        "repo": r.get("repo") or repo_token,
                        "strategy": strategy,
                    })
                    submitted += 1
                    if limit and limit > 0 and submitted >= limit:
                        stop = True
                        break
                _flush(batch)
                logger.info("Progress: converted=%d skipped=%d (joined=%d)",
                            converted, skipped, submitted)
                if stop:
                    break

        logger.info(
            "Done joining. converted=%d skipped=%d; dropped %d not-in-hamishivi, "
            "%d empty-patch, %d empty-FAIL_TO_PASS.",
            converted, skipped, n_not_in_ham, n_empty_patch, n_empty_f2p,
        )
        return converted, skipped

    # -- Convert one joined row --------------------------------------------
    def convert_row(self, row: Dict[str, Any], dst_root: Path) -> Optional[str]:
        task_id = row["task_id"]
        image = row["image"]
        bug_patch = row["bug_patch"]
        f2p = row["fail_to_pass"]
        p2p = row["pass_to_pass"]
        if not image or not bug_patch or not f2p:
            return None

        task_name = _sanitize_name(task_id)
        out = dst_root / task_name
        out.mkdir(parents=True, exist_ok=True)

        # Materialize the bug patch beside the def; %files addresses it by an
        # absolute path so `apptainer build` works regardless of invocation CWD.
        # Ensure a trailing newline so `git apply`/`patch` accept the last hunk.
        bug_path = out / "bug.patch"
        patch_text = bug_patch if bug_patch.endswith("\n") else bug_patch + "\n"
        bug_path.write_text(patch_text)

        enriched: Dict[str, Any] = {
            "name": task_name,
            # Native taxonomy left empty; the downstream LLM classifier fills
            # the classified_* fields used by the composition module.
            "domain": "unknown",
            "skill_type": "unknown",
            "primitive_skills": [],
            "task_complexity": "unknown",
            "command_complexity": "unknown",
            "scenario": "",
            "language": "any (model's choice)",
            "description": row["instruction"],
            "truth": "",
            # Dataset-native metadata preserved for the appendix panels.
            "swe_smith_image": image,
            "swe_smith_repo": row.get("repo", ""),
            "swe_smith_strategy": row.get("strategy", ""),
            "swe_smith_num_fail_to_pass": len(f2p),
            "swe_smith_num_pass_to_pass": len(p2p),
            "swe_smith_fail_to_pass": f2p,
            # Provenance.
            "source": "swe_smith",
            "source_repo": HF_REPO_ID,
            "source_verifier_repo": VERIFIER_REPO_ID,
            "source_slug": task_id,
        }
        (out / "task.json").write_text(json.dumps(enriched, indent=2))

        (out / "container.def").write_text(_build_container_def(
            base_image=image,
            bug_patch_abs=bug_path.resolve(),
        ))

        wrapper = _TEST_FINAL_WRAPPER.replace(
            "{F2P_JSON}", json.dumps(f2p)
        ).replace(
            "{P2P_JSON}", json.dumps(p2p)
        )
        (out / "test_final_state.py").write_text(wrapper)
        (out / "test_initial_state.py").write_text(_PLACEHOLDER_INITIAL_STATE)

        return task_name

register_adapter(SweSmithAdapter())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_swe_smith_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(SweSmithAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true",
                    help="Reuse the HF datasets cache instead of re-fetching "
                         "(load_dataset is already cache-aware; this is a no-op "
                         "hint kept for parity with the other adapters).")
    ap.add_argument("--revision", type=str, default=None,
                    help="hamishivi/agent-task-swe-smith revision (commit SHA) to pin.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = SweSmithAdapter()

    converted, skipped = adapter.convert_streaming(
        args.dst.resolve(),
        limit=args.limit,
        workers=args.workers,
        revision=args.revision,
        cache_dir=args.cache_dir.resolve(),
    )
    logger.info("Done. converted=%d skipped=%d  dst=%s",
                converted, skipped, args.dst)


if __name__ == "__main__":
    main()
