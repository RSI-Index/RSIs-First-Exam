"""Adapter for TermiGen's ``ucsb-mlsec/terminal-bench-env`` (Harbor 2.0 format).

Source: https://github.com/ucsb-mlsec/terminal-bench-env
(paper: "TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis
for Terminal Agents", arXiv:2602.07274).

The dataset ships >3500 Harbor-style task directories under
``environments_harbor/`` in that GitHub repo.  Each task has exactly the
same layout the ``endless-terminals`` / ``OpenThoughts-TB`` adapters already
know how to digest:

    task_name/
      task.toml                 # author, category, difficulty, tags, ...
      instruction.md            # natural-language prompt
      environment/
        Dockerfile              # FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:...
        [data/ ...]             # payload files referenced by COPY
      tests/
        test.sh                 # wraps `pytest /tests/test_outputs.py`
        test_outputs.py         # the real verifier (we run it directly)

Two wrinkles vs. the existing adapters:

1.  Source is GitHub, not the HF Hub.  We can't use
    ``huggingface_hub.snapshot_download``; instead we do a **partial clone**
    + **sparse checkout** of the ``environments_harbor/`` subtree so we skip
    the large ``termigen_env.zip`` (the TB-1.0 artifact we don't use) and
    save ~200 MB of wire traffic.

2.  Dockerfiles use a Terminal-Bench–specific base image
    (``ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624``).  Letting
    each of 3,500 per-task builds re-pull that base from ghcr.io is slow
    and brittle.  Our solve script (``run_generate_solutions_termigen.sh``)
    pre-bakes a shared local base SIF at
    ``$PROJECT_ROOT/tbench_ubuntu24_base.sif``; after
    :func:`flatten_harbor_task` derives a ``container.def`` from the
    Dockerfile, we rewrite its ``Bootstrap: docker`` / ``From: ghcr.io/...``
    header to ``Bootstrap: localimage`` / ``From: ./tbench_ubuntu24_base.sif``
    so per-task builds layer on top of the prebaked image.  See
    :func:`_rewrite_container_def_to_localimage`.

CLI:

    python -m rl_data.comparison.adapters.termigen \\
        --dst rl_data/output/tasks_termigen

Options mirror the other adapters (``--limit``, ``--workers``,
``--skip-download``, ``--revision`` for a specific commit SHA).
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rl_data.comparison.adapters import (
    Adapter,
    flatten_harbor_task,
    register_adapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default parameters — tweak here (or via env) without touching call sites.
# ---------------------------------------------------------------------------

# Upstream repo and the subdir that holds the Harbor-format task dirs.
GITHUB_REPO_URL = "https://github.com/ucsb-mlsec/terminal-bench-env.git"
HARBOR_SUBDIR = "environments_harbor"

# ghcr.io base image referenced by every TermiGen Dockerfile. If upstream bumps
# this tag we need to update both here *and* the tbench base SIF recipe in
# ``scripts/comparison/run_generate_solutions_termigen.sh``.
TBENCH_BASE_IMAGE = "ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"

# Per-task container.defs switch their Bootstrap/From to this local SIF so
# every task reuses a single prebuilt base and per-task builds stay cheap.
# Relative path — resolved against CWD at apptainer-build time (PROJECT_ROOT).
LOCAL_BASE_SIF = "./tbench_ubuntu24_base.sif"


# ---------------------------------------------------------------------------
# Partial-clone + sparse-checkout helper
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """Run a subprocess command, streaming its output into our logger."""
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _sparse_clone_or_update(repo_url: str, dest: Path, *, revision: Optional[str]) -> None:
    """Idempotently clone ``repo_url`` into ``dest`` with a sparse checkout
    of the Harbor tasks subtree only.

    Uses ``--filter=blob:none`` so we don't pull blob history for files we
    never touch, and ``sparse-checkout set`` so only ``environments_harbor``
    is materialized on disk. Subsequent runs fast-forward.

    ``revision`` pins a commit SHA / tag; when ``None`` we track ``main``.
    """
    if (dest / ".git").exists():
        logger.info("Reusing existing clone at %s", dest)
        # Make sure the sparse filter still matches what we want (in case an
        # earlier run set it differently).
        try:
            _run(["git", "sparse-checkout", "set", HARBOR_SUBDIR], cwd=dest)
            _run(["git", "fetch", "--filter=blob:none", "origin"], cwd=dest)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"git fetch failed in {dest}: {exc}") from exc
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run([
                "git", "clone",
                "--filter=blob:none",
                "--sparse",
                "--no-checkout",
                repo_url,
                str(dest),
            ])
            _run(["git", "sparse-checkout", "init", "--cone"], cwd=dest)
            _run(["git", "sparse-checkout", "set", HARBOR_SUBDIR], cwd=dest)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"git clone + sparse-checkout of {repo_url} failed: {exc}"
            ) from exc

    # Pin to the requested revision (default: latest main).
    target = revision if revision else "origin/main"
    try:
        _run(["git", "checkout", target], cwd=dest)
    except subprocess.CalledProcessError:
        # Maybe ``target`` is a tag/SHA that is reachable but not yet fetched
        # with the shallow filter; re-fetch everything by SHA and retry.
        _run(["git", "fetch", "--filter=blob:none", "origin", target], cwd=dest)
        _run(["git", "checkout", target], cwd=dest)


# ---------------------------------------------------------------------------
# Per-task def rewrite: ghcr.io → local base SIF
# ---------------------------------------------------------------------------


_BOOTSTRAP_RE = re.compile(r"^\s*Bootstrap\s*:\s*\S+\s*$", re.IGNORECASE | re.MULTILINE)
_FROM_RE = re.compile(r"^\s*From\s*:\s*\S+.*$", re.IGNORECASE | re.MULTILINE)


def _rewrite_container_def_to_localimage(def_path: Path) -> None:
    """Rewrite the ``Bootstrap:``/``From:`` header of a derived def to layer on
    top of the prebaked ``tbench_ubuntu24_base.sif`` instead of pulling the
    ghcr.io base on every per-task build.

    Idempotent: running twice is a no-op after the first pass.
    """
    try:
        text = def_path.read_text()
    except OSError as exc:
        logger.warning("rewrite_container_def: cannot read %s: %s", def_path, exc)
        return

    new_bootstrap = "Bootstrap: localimage"
    new_from = f"From: {LOCAL_BASE_SIF}"

    replaced_bootstrap, n_boot = _BOOTSTRAP_RE.subn(new_bootstrap, text, count=1)
    if n_boot == 0:
        # Unusual shape (no Bootstrap line at all) — fall back to prepending
        # a fresh header so the resulting file still builds.
        replaced_bootstrap = new_bootstrap + "\n" + new_from + "\n\n" + text
        def_path.write_text(replaced_bootstrap)
        return

    replaced, n_from = _FROM_RE.subn(new_from, replaced_bootstrap, count=1)
    if n_from == 0:
        # Bootstrap was present but no From — inject one right after Bootstrap.
        replaced = _BOOTSTRAP_RE.sub(
            new_bootstrap + "\n" + new_from, replaced_bootstrap, count=1,
        )

    def_path.write_text(replaced)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TermiGenAdapter(Adapter):
    """TermiGen / ``ucsb-mlsec/terminal-bench-env`` (Harbor 2.0 format)."""

    name = "termigen"
    # Not on HF Hub; we override ``fetch`` to git-clone instead.
    hf_repo_id = None
    default_dst = "rl_data/output/tasks_termigen"

    # -- Fetch -------------------------------------------------------------
    def fetch(self, cache_dir: Path, *, revision: Optional[str] = None) -> Path:
        """Sparse-clone the ``environments_harbor/`` subtree of the upstream
        GitHub repo into ``cache_dir/repo`` and return the absolute path of
        the resulting task directory root.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = (cache_dir / "repo").resolve()
        _sparse_clone_or_update(GITHUB_REPO_URL, repo_dir, revision=revision)

        harbor_root = repo_dir / HARBOR_SUBDIR
        if not harbor_root.is_dir():
            raise RuntimeError(
                f"Sparse clone succeeded but {harbor_root} is missing; the "
                f"upstream layout may have changed."
            )
        return harbor_root

    # -- Iterate source task dirs ------------------------------------------
    # The default implementation walks ``snapshot_dir`` directly which is
    # exactly what we want (``harbor_root`` is already ``environments_harbor/``).

    # -- Convert one task --------------------------------------------------
    def convert_one(self, src: Path, dst_root: Path) -> Optional[str]:
        task_name = flatten_harbor_task(
            src, dst_root,
            source_name="tg",
            source_repo="ucsb-mlsec/terminal-bench-env",
            prefix="tg_",
            # Harbor 2.0 ships `tests/test.sh` (wraps `pytest test_outputs.py`)
            # and the real `tests/test_outputs.py`.  We run test_outputs.py
            # directly through our pytest harness -- the shell wrapper only
            # exists so Harbor's agent runner can surface a reward line, which
            # we don't consume.
            test_final_candidates=("test_outputs.py",),
            # Some test_outputs.py read sibling data files (CSV fixtures,
            # reference outputs, etc.); mirror OT-TB's behaviour and copy
            # everything non-shell from tests/.
            copy_aux_test_files=True,
        )
        if task_name is None:
            return None

        # Swap the derived def's Bootstrap/From header so the per-task builds
        # layer on the shared prebaked base SIF (see module docstring).
        _rewrite_container_def_to_localimage(dst_root / task_name / "container.def")
        return task_name


register_adapter(TermiGenAdapter())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_termigen_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(TermiGenAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true",
                    help="Reuse an existing sparse clone in cache-dir instead "
                         "of fetching from GitHub.")
    ap.add_argument("--revision", type=str, default=None,
                    help="Commit SHA or tag to pin (default: origin/main).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = TermiGenAdapter()

    if args.skip_download:
        snapshot = (args.cache_dir / "repo" / HARBOR_SUBDIR).resolve()
        logger.info("Skipping download; using %s", snapshot)
        if not snapshot.exists():
            logger.error(
                "Sparse clone at %s does not exist yet; cannot --skip-download "
                "on a cold cache. Run once without the flag first.", snapshot,
            )
            raise SystemExit(1)
    else:
        snapshot = adapter.fetch(args.cache_dir.resolve(), revision=args.revision)

    converted, skipped = adapter.convert_all(
        snapshot, args.dst.resolve(),
        limit=args.limit, workers=args.workers,
    )
    logger.info("Done. converted=%d skipped=%d  dst=%s",
                converted, skipped, args.dst)


if __name__ == "__main__":
    main()
