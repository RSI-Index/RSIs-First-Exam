"""Adapter for ``hamishivi/agent-task-cli-gym`` (CLI-Gym, 1,552 tasks).

CLI-Gym (``arXiv:2602.10999``, LiberCoders/CLI-Gym) derives *environment-
intensive* terminal tasks by **environment inversion**: starting from a
SWE-Smith gold instance (a real Python repo installed at ``/testbed`` in a
conda env named ``testbed``, with a passing unit-test suite), an agent is
asked to deliberately *corrupt the environment* (e.g. swap glibc locale
``language``/``territory`` fields, poison a codec registry, truncate a shared
library) so that a chosen subset of unit tests starts failing. The published
task is the *repair* problem: the agent is dropped into the corrupted image
and must restore the environment so those unit tests pass again.

Two HF sources, joined on ``task_id``
-------------------------------------
The dataset the comparison consumes (``hamishivi/agent-task-cli-gym``) is the
open-instruct RLVR packaging: each parquet row carries the *pre-built* faulty
Docker image and a chat-formatted prompt, but **not** the verifier:

    messages     [system, user]   # user msg embeds the vanillux workflow boilerplate
    ground_truth <task_id slug>
    dataset      "passthrough"     # reward is env-computed (PassthroughVerifier)
    env_config   {env_name: "swerl_vanillux_sandbox",
                  image:    "hamishi740/agent-task-cli-gym:<12-hex>",
                  task_id:  "<slug>"}
    source       "cli_gym"

The authoritative verifier lives in the upstream release
(``LiberCoders/CLI-Gym``, 1,655 rows), keyed by the same ``task_id``:

    task_id        <slug>
    task_yaml      # YAML with the CLEAN instruction (no harness boilerplate)
    dockerfile     # FROM jyangballin/swesmith.x86_64.<repo>.<sha> + corruption RUNs
    docker_compose # t-bench scaffolding (unused by us)
    run_tests      # the run-tests.sh: `conda activate testbed; pytest <selected UTs>`

All 1,552 ``hamishivi`` task_ids are a subset of the 1,655 ``LiberCoders``
task_ids, so the adapter loads both, joins on ``task_id``, and pairs every
pre-built image with its selected-unit-test list.

The task **instruction** is recovered from the hamishivi user message (stripped
of the vanillux workflow boilerplate our own harness re-adds). That text is
byte-identical to the upstream ``task_yaml`` ``instruction`` wherever the YAML
parses, and is always present, so we avoid parsing the ~6 % of ``task_yaml``
files whose ``instruction: |`` block scalar uses inconsistent indentation
(malformed YAML); the ``task_yaml`` is only a never-hit fallback.

Verifier contract
------------------
Upstream grading (the commented-out ``parser.py`` in ``run_tests``) runs the
selected unit tests in the ``testbed`` conda env and marks the task *resolved*
iff **all** of them pass (SWE-bench ``ResolvedStatus.FULL``). Running
``pytest <selected UTs>`` and checking the exit code is exactly equivalent:
pytest exits non-zero iff any selected test fails or errors. So our
``test_final_state.py`` shells out to::

    source /opt/miniconda3/bin/activate; conda activate testbed; cd /testbed
    pytest --disable-warnings --color=no --tb=short <UT1> <UT2> ...

and asserts ``returncode == 0``.

The agent's environment fixes persist because the solve harness keeps a single
``--writable-tmpfs`` Apptainer instance alive across the rollout *and* the
final test (system files under ``/usr/share/i18n``, ``/testbed``, the conda env,
etc. are all writable in the overlay and visible to the verifier).

Wrinkles vs. the other adapters
-------------------------------
1.  **Two-source join, not a tarball/snapshot.** Unlike R2E Gym / TerminalTraj
    (single HF tarball) this adapter reads two parquet datasets via
    ``datasets.load_dataset`` and joins them in memory; there are no source
    *directories* to walk, so we override the convert pipeline.

2.  **Per-task pre-built Docker Hub image.** Every task ``FROM``s its own
    ``hamishi740/agent-task-cli-gym:<hash>`` image (~900 MB each, all distinct,
    public on Docker Hub). Like R2E Gym / TerminalTraj we cannot pre-bake a
    shared base SIF; the solve script's pre-build phase pulls each one.

3.  **Outer-harness pytest must live on the base-conda PATH.** The solve
    harness runs ``pytest pytest_final_state.py`` from the container's default
    (non-login) shell, whose PATH leads with ``/opt/miniconda3/bin`` — the
    *base* conda env, which ships no pytest. The real unit tests run inside the
    ``testbed`` env (which has its own pytest + the project installed). So the
    container.def's ``%post`` makes a pytest available on the base PATH (pip
    install into base; falling back to symlinking the testbed pytest) purely so
    the verifier *wrapper* can be collected — the wrapper itself activates
    ``testbed`` for the actual run.

4.  **Whole-suite tasks are skipped.** ~6 % of CLI-Gym ``run_tests`` scripts
    select *no* explicit tests (the ``pytest`` invocation has no targets → run
    the entire suite). Requiring the full upstream suite to pass is not a fair
    fail-to-pass verifier (it folds in unrelated xfails/flakes), so the adapter
    skips those tasks rather than ingest an unverifiable env.

CLI:

    python -m rl_data.comparison.adapters.cli_gym \\
        --dst rl_data/output/tasks_cli_gym

Options mirror the other adapters (``--limit``, ``--workers``,
``--skip-download``, ``--revision``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rl_data.comparison.adapters import (
    Adapter,
    _PLACEHOLDER_INITIAL_STATE,
    register_adapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Pre-built faulty images + open-instruct RLVR prompts (the dataset the
# comparison is asked to integrate).
HF_REPO_ID = "hamishivi/agent-task-cli-gym"
# Upstream release carrying the authoritative verifier (run-tests.sh) +
# clean instruction, keyed by the same task_id.
VERIFIER_REPO_ID = "LiberCoders/CLI-Gym"

# The uniform pytest invocation every CLI-Gym run_tests.sh uses, sans targets.
_PYTEST_PREFIX = (
    "source /opt/miniconda3/bin/activate; conda activate testbed; "
    "pytest --disable-warnings --color=no --tb=no --verbose"
)


# ---------------------------------------------------------------------------
# %post: make a pytest reachable on the BASE-conda PATH for the outer wrapper.
# The real tests run in the `testbed` env (see _TEST_FINAL_WRAPPER); this is
# only so the harness's `pytest pytest_final_state.py` collection succeeds.
# Fail-soft (set +e) and ends with the chmod /home/user convention that
# generate_solutions._patch_def_chmod expects.
# ---------------------------------------------------------------------------
_PYTEST_BASE_BOOTSTRAP_POST = r"""    set +e
    # --- CLI-Gym outer-harness pytest bootstrap ---------------------------
    # The solve harness runs `pytest pytest_final_state.py` from the default
    # (base-conda) PATH; base conda ships no pytest. Install one there so the
    # verifier wrapper can be collected. The wrapper re-activates the `testbed`
    # env for the real unit-test run, so this base pytest never touches the
    # project's own dependencies.
    if ! /opt/miniconda3/bin/python -c 'import pytest' >/dev/null 2>&1; then
        /opt/miniconda3/bin/python -m pip install --no-cache-dir pytest >/dev/null 2>&1
    fi
    if ! /opt/miniconda3/bin/python -c 'import pytest' >/dev/null 2>&1; then
        # Fallback: expose the testbed env's pytest on the base PATH. Its
        # shebang points at the testbed python, which can still import+run our
        # dependency-free wrapper.
        ln -sf /opt/miniconda3/envs/testbed/bin/pytest /opt/miniconda3/bin/pytest 2>/dev/null || true
    fi
    set -e
    mkdir -p /home/user
    chmod 755 /home/user
"""


# ---------------------------------------------------------------------------
# Pytest verifier wrapper (test_final_state.py).
#
# Shells out to the testbed-env pytest over the task's selected unit tests and
# asserts they all pass. ``{TEST_TARGETS_JSON}`` is substituted at ingest with
# a JSON list of pytest node ids / file paths.
# ---------------------------------------------------------------------------
_TEST_FINAL_WRAPPER = r'''"""Pytest wrapper around CLI-Gym's selected-unit-test verifier.

CLI-Gym tasks are SWE-Smith repos (installed at ``/testbed`` in the ``testbed``
conda env) whose environment has been deliberately corrupted so a chosen subset
of unit tests fails. The task is resolved iff, after the agent's repair, **all**
of those selected tests pass again (SWE-bench ``ResolvedStatus.FULL``).

We replicate the upstream run-tests.sh command exactly — activate the ``testbed``
conda env and run ``pytest`` over the selected node ids — and surface the outcome
to our pytest-based harness via the process exit code (pytest exits non-zero iff
any selected test fails or errors).
"""
import json
import shlex
import subprocess

# Selected unit tests for this task (CLI-Gym fail-to-pass + pass-to-pass set).
TEST_TARGETS = json.loads(r"""{TEST_TARGETS_JSON}""")

_ACTIVATE = "source /opt/miniconda3/bin/activate; conda activate testbed"
_PYTEST_FLAGS = "--disable-warnings --color=no --tb=short -p no:cacheprovider"


def test_cli_gym_verifier():
    assert TEST_TARGETS, "no selected unit tests baked into this task"
    targets = " ".join(shlex.quote(t) for t in TEST_TARGETS)
    cmd = f"{_ACTIVATE}; cd /testbed; pytest {_PYTEST_FLAGS} {targets}"
    proc = subprocess.run(
        ["bash", "-c", cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "CLI-Gym verifier: not all selected unit tests passed "
        f"(pytest exit={proc.returncode}, {len(TEST_TARGETS)} target(s)).\n"
        "stdout tail:\n" + (proc.stdout or "")[-3000:] + "\n"
        "stderr tail:\n" + (proc.stderr or "")[-1000:]
    )
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Where the hamishivi user-message instruction stops and the vanillux harness
# boilerplate (which our own harness re-adds) begins. Used only as a fallback
# when the LiberCoders task.yaml instruction is unavailable.
_HAM_BOILERPLATE_MARKER = "\n\nYou can execute bash commands"
_HAM_LEADIN = "Please solve this task:\n\n"


def _extract_test_targets(run_tests: str) -> List[str]:
    """Pull the pytest target list out of a CLI-Gym ``run_tests`` script.

    Every script wraps a single-quoted command string passed to
    ``run_and_log('source ...; conda activate testbed; pytest <flags> <targets>',
    "/test.log")``. We locate that command, slice off everything through the
    fixed pytest-flags prefix, and return the remaining whitespace-split
    targets. Returns ``[]`` when the script selects no explicit tests (a
    whole-suite run we deliberately skip upstream).
    """
    anchor = run_tests.find("'source /opt/miniconda3/bin/activate")
    if anchor < 0:
        return []
    start = anchor + 1
    end = run_tests.find("'", start)
    if end < 0:
        return []
    cmd = run_tests[start:end]
    idx = cmd.find(_PYTEST_PREFIX)
    if idx < 0:
        # Defensive: tolerate minor flag reorderings by anchoring on "pytest ".
        m = re.search(r"conda activate testbed;\s*pytest\b[^\n]*?--verbose\s*", cmd)
        if not m:
            return []
        targets_str = cmd[m.end():]
    else:
        targets_str = cmd[idx + len(_PYTEST_PREFIX):]
    return targets_str.split()


_SWESMITH_FROM_RE = re.compile(
    r"FROM\s+(\S*swesmith[^\s.]*\.[^\s.]+\.([^\s.]+)\.[0-9a-f]+)", re.IGNORECASE
)


def _parse_swesmith_repo(dockerfile: str) -> Tuple[str, str]:
    """Best-effort (full base image, repo token) from a CLI-Gym Dockerfile.

    ``FROM jyangballin/swesmith.x86_64.joke2k_1776_faker.8b401a7d`` ->
    (``jyangballin/swesmith.x86_64.joke2k_1776_faker.8b401a7d``,
     ``joke2k_1776_faker``). Returns ("", "") if no swesmith FROM is found.
    """
    if not dockerfile:
        return "", ""
    m = _SWESMITH_FROM_RE.search(dockerfile)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def _instruction_from_yaml(task_yaml: str) -> str:
    """Parse the clean ``instruction:`` field from a CLI-Gym task.yaml."""
    if not task_yaml:
        return ""
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("PyYAML not installed; falling back to message instruction")
        return ""
    try:
        data = yaml.safe_load(task_yaml) or {}
    except Exception as exc:  # yaml.YAMLError etc.
        logger.warning("task.yaml parse failed: %s", exc)
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("instruction") or "").strip()


def _instruction_from_messages(messages: Any) -> str:
    """Fallback: recover the core instruction from the hamishivi chat row.

    The user message is ``"Please solve this task:\\n\\n<instruction>\\n\\nYou
    can execute bash commands ...<vanillux boilerplate>"``. Our own vanillux
    harness re-adds that boilerplate, so we strip it back down to the core
    instruction to avoid duplicating the workflow scaffolding in the prompt.
    """
    if not messages:
        return ""
    user = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            user = str(m.get("content") or "")
            break
    if not user:
        return ""
    if user.startswith(_HAM_LEADIN):
        user = user[len(_HAM_LEADIN):]
    cut = user.find(_HAM_BOILERPLATE_MARKER)
    if cut >= 0:
        user = user[:cut]
    return user.strip()


def _build_container_def(*, base_image: str) -> str:
    """Apptainer container.def for one CLI-Gym task.

    Bootstraps from the pre-built faulty Docker Hub image and only layers the
    outer-harness pytest bootstrap; the unit tests, conda env and repo are all
    already baked into the image.
    """
    lines: List[str] = [
        "Bootstrap: docker",
        f"From: {base_image}",
        "",
        "%post",
        _PYTEST_BASE_BOOTSTRAP_POST.rstrip(),
        "",
        "%labels",
        "    Author cli-gym-adapter",
        f"    BaseImage {base_image}",
        '    Description "CLI-Gym faulty-environment repair task (SWE-Smith testbed)"',
        "",
    ]
    return "\n".join(lines) + "\n"


def _sanitize_name(task_id: str) -> str:
    """``cligym_`` + slug, with whitespace/path separators normalized."""
    slug = re.sub(r"\s+", "_", task_id.strip())
    slug = slug.replace("/", "_")
    return "cligym_" + slug


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CliGymAdapter(Adapter):
    """``hamishivi/agent-task-cli-gym`` (pre-built faulty images + SWE-Smith UTs).

    Two-source adapter: images/prompts from ``hamishivi/agent-task-cli-gym``,
    verifiers/instructions from ``LiberCoders/CLI-Gym``, joined on ``task_id``.
    """

    name = "cli_gym"
    hf_repo_id = HF_REPO_ID
    default_dst = "rl_data/output/tasks_cli_gym"

    # The base-class dir-walking convert pipeline does not apply (the source is
    # two parquet tables, not task directories). We override with a row-based
    # flow below; ``convert_one`` stays unused.
    def convert_one(self, src: Path, dst_root: Path) -> Optional[str]:  # pragma: no cover
        raise NotImplementedError("CliGymAdapter converts joined rows, not dirs")

    # -- Load + join both parquet sources ----------------------------------
    def load_joined_rows(
        self,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Load both HF datasets and join on ``task_id``.

        Returns one dict per joinable task with keys: ``task_id``, ``image``,
        ``instruction``, ``test_targets``, ``base_image``, ``swesmith_repo``.
        Tasks present in hamishivi but missing a LiberCoders verifier, or with
        an empty (whole-suite) test selection, are dropped with a log line.
        """
        from datasets import load_dataset

        cache_str = str(cache_dir) if cache_dir else None

        logger.info("Loading %s (images + prompts) ...", self.hf_repo_id)
        ham = load_dataset(
            self.hf_repo_id, split="train", revision=revision, cache_dir=cache_str
        )
        logger.info("Loading %s (verifiers) ...", VERIFIER_REPO_ID)
        lib = load_dataset(VERIFIER_REPO_ID, split="train", cache_dir=cache_str)

        # Index the verifier source by task_id.
        verifiers: Dict[str, Dict[str, Any]] = {}
        for r in lib:
            verifiers[r["task_id"]] = r

        rows: List[Dict[str, Any]] = []
        n_missing_verifier = 0
        n_empty_tests = 0
        for r in ham:
            env_cfg = r.get("env_config") or {}
            task_id = env_cfg.get("task_id") or r.get("ground_truth")
            image = env_cfg.get("image")
            if not task_id or not image:
                continue
            v = verifiers.get(task_id)
            if v is None:
                n_missing_verifier += 1
                continue
            targets = _extract_test_targets(v.get("run_tests") or "")
            if not targets:
                n_empty_tests += 1
                continue
            # Instruction source: prefer the hamishivi chat row. It carries the
            # same problem statement as the upstream task.yaml (verified
            # byte-identical wherever the YAML parses) and is always present and
            # clean, so we don't need to parse the task.yaml at all -- which also
            # keeps us off the malformed-block-scalar warning path for the ~6 %
            # of tasks whose ``instruction: |`` block uses inconsistent
            # indentation. The task.yaml is only a (never-observed) fallback.
            instruction = _instruction_from_messages(r.get("messages"))
            if not instruction:
                instruction = _instruction_from_yaml(v.get("task_yaml") or "")
            base_image, swesmith_repo = _parse_swesmith_repo(v.get("dockerfile") or "")
            rows.append({
                "task_id": task_id,
                "image": image,
                "instruction": instruction,
                "test_targets": targets,
                "base_image": base_image,
                "swesmith_repo": swesmith_repo,
            })

        logger.info(
            "Joined %d task(s); dropped %d missing-verifier, %d whole-suite "
            "(no explicit tests).",
            len(rows), n_missing_verifier, n_empty_tests,
        )
        return rows

    # -- Convert one joined row --------------------------------------------
    def convert_row(self, row: Dict[str, Any], dst_root: Path) -> Optional[str]:
        task_id = row["task_id"]
        image = row["image"]
        targets = row["test_targets"]
        if not image or not targets:
            return None

        task_name = _sanitize_name(task_id)
        out = dst_root / task_name
        out.mkdir(parents=True, exist_ok=True)

        enriched: Dict[str, Any] = {
            "name": task_name,
            # Native taxonomy left empty; the downstream LLM classifier fills
            # the classified_* fields used by the composition module. CLI-Gym
            # ships no domain/difficulty metadata of its own.
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
            "cli_gym_image": image,
            "cli_gym_base_image": row.get("base_image", ""),
            "cli_gym_swesmith_repo": row.get("swesmith_repo", ""),
            "cli_gym_num_tests": len(targets),
            "cli_gym_test_targets": targets,
            # Provenance.
            "source": "cli_gym",
            "source_repo": HF_REPO_ID,
            "source_verifier_repo": VERIFIER_REPO_ID,
            "source_slug": task_id,
        }
        (out / "task.json").write_text(json.dumps(enriched, indent=2))

        (out / "container.def").write_text(_build_container_def(base_image=image))

        wrapper = _TEST_FINAL_WRAPPER.replace(
            "{TEST_TARGETS_JSON}", json.dumps(targets)
        )
        (out / "test_final_state.py").write_text(wrapper)
        (out / "test_initial_state.py").write_text(_PLACEHOLDER_INITIAL_STATE)

        return task_name

    # -- Bulk convert joined rows ------------------------------------------
    def convert_rows(
        self,
        rows: List[Dict[str, Any]],
        dst_root: Path,
        *,
        limit: int = 0,
        workers: int = 16,
    ) -> Tuple[int, int]:
        if limit and limit > 0:
            rows = rows[:limit]
        dst_root.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.convert_row, row, dst_root): row for row in rows}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    logger.warning(
                        "convert_row failed on %s: %s",
                        futs[fut].get("task_id"), e,
                    )
                    r = None
                if r is not None:
                    converted += 1
                else:
                    skipped += 1
                total = converted + skipped
                if total % 500 == 0:
                    logger.info("Progress: %d/%d (converted=%d, skipped=%d)",
                                total, len(rows), converted, skipped)
        return converted, skipped


register_adapter(CliGymAdapter())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_cli_gym_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(CliGymAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true",
                    help="Reuse the HF datasets cache instead of re-fetching "
                         "(load_dataset is already cache-aware; this is a no-op "
                         "hint kept for parity with the other adapters).")
    ap.add_argument("--revision", type=str, default=None,
                    help="hamishivi/agent-task-cli-gym revision (commit SHA) to pin.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = CliGymAdapter()

    rows = adapter.load_joined_rows(
        revision=args.revision,
        cache_dir=args.cache_dir.resolve(),
    )
    converted, skipped = adapter.convert_rows(
        rows, args.dst.resolve(),
        limit=args.limit, workers=args.workers,
    )
    logger.info("Done. converted=%d skipped=%d  dst=%s",
                converted, skipped, args.dst)


if __name__ == "__main__":
    main()
