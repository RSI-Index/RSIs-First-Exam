"""Adapter for TerminalTraj (``m-a-p/TerminalTraj-5k-instances``).

Source: https://github.com/multimodal-art-projection/TerminalTraj
(paper: "TerminalTraj: Large-Scale Terminal Agentic Trajectory Generation
from Dockerized Environments", arXiv:2602.01244).

The "5k instances" release ships 5,660 tasks as a single ~13 MB tarball
(``5k_instances.tar.gz``) on the HF Hub. Each task follows TerminalBench's
v1.0 layout (**not** Harbor 2.0), so ``flatten_harbor_task`` doesn't apply
directly:

    task_<N>/
      Dockerfile              # single `FROM yizhilll/tb_container-<hash>:tmux_asciinema_v2`
      docker-compose.yaml     # t-bench orchestration scaffolding (unused by us)
      item_info.json          # upstream repo provenance (GitHub URL, etc.)
      run-tests.sh            # uv+pip boilerplate that wraps pytest (unused by us)
      task.yaml               # instruction + author + difficulty/category/tags (YAML!)
      tests/
        test_outputs.py       # the real pytest verifier

Three non-trivial wrinkles vs. ET / OpenThoughts / TermiGen:

1.  **Source is a single tarball on HF.** We ``hf_hub_download`` just
    ``5k_instances.tar.gz`` (13 MB) and extract it into the cache dir.
    No snapshot_download of the full repo — it only ships this one file
    anyway plus ``.gitattributes``.

2.  **Every task has a unique Docker Hub base image.** The 5,660 images
    are all distinct (``yizhilll/tb_container-<md5>:tmux_asciinema_v2``),
    each ~400 MB, because each was built from a different upstream GitHub
    repo. We cannot pre-bake a single shared base SIF like we do for
    TermiGen — per-task Docker pulls are unavoidable. The images are
    public on Docker Hub, pullable anonymously (but creds help vs. rate
    limits).

3.  **Base images span many distros and are missing pytest.** The
    images inherit from whatever the original repo's Dockerfile produced
    (observed: Debian bookworm, Fedora 27, Ubuntu, Alpine, …), with old
    or missing ``pip3``. We inject a robust pytest-bootstrap ``%post``
    snippet that tries, in order: (a) existing pip3, (b) system package
    manager (apt/dnf/yum/apk), (c) ``get-pip.py``. Anything that can't
    be bootstrapped is a genuine build failure we want surfaced rather
    than silently tolerated.

Additionally, **some base images use glibc older than the fakeroot binary
bundled with Apptainer (e.g. the Fedora 27 tasks)**, so builds must pass
``--ignore-fakeroot-command``. Our solve script handles that in the
pre-build phase; the generated ``container.def`` itself is agnostic.

CLI:

    python -m rl_data.comparison.adapters.terminaltraj \\
        --dst rl_data/output/tasks_terminaltraj

Options mirror the other adapters (``--limit``, ``--workers``,
``--skip-download``, ``--revision`` for a specific HF revision).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

from rl_data.comparison.adapters import (
    Adapter,
    _PLACEHOLDER_INITIAL_STATE,
    _dockerfile_to_apptainer_def,
    register_adapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

HF_REPO_ID = "m-a-p/TerminalTraj-5k-instances"
TARBALL_NAME = "5k_instances.tar.gz"
# Top-level directory created when the tarball is extracted.
EXTRACTED_ROOT_NAME = "5k_instances"


# Robust pytest bootstrap appended to every task's container.def's %post.
# See module docstring for why this is needed. The three-step fallback chain
# is deliberately fail-soft: ``set +e`` around it keeps the build going so a
# late step can still succeed even if an earlier one hits a hiccup. The final
# ``chmod /home/user`` lines are the tmax convention that generate_solutions's
# _patch_def_chmod expects.
_PYTEST_BOOTSTRAP_POST = r"""    set +e
    # --- TerminalTraj pytest bootstrap -----------------------------------
    # TerminalTraj base images span many distros (Debian/Ubuntu/Fedora/Alpine
    # /...) and most ship without pytest (upstream's run-tests.sh installs it
    # via uv at test time -- we instead bake it into the SIF so our harness's
    # `apptainer exec <sif> pytest ...` works). Try the shortest paths first.
    # 1) Existing pip3 (commonly present on Debian/Ubuntu/Fedora bases).
    if command -v pip3 >/dev/null 2>&1; then
        pip3 install --break-system-packages --no-cache-dir pytest >/dev/null 2>&1 \
          || pip3 install --no-cache-dir pytest >/dev/null 2>&1
    fi
    # 2) Install pip via system package manager, then retry.
    if ! python3 -c 'import pytest' >/dev/null 2>&1; then
        (command -v apt-get >/dev/null 2>&1 && apt-get update -qq \
              && apt-get install -y -qq python3-pip) >/dev/null 2>&1
        (command -v dnf >/dev/null 2>&1 && dnf install -y -q python3-pip) >/dev/null 2>&1
        (command -v yum >/dev/null 2>&1 && yum install -y -q python3-pip) >/dev/null 2>&1
        (command -v apk >/dev/null 2>&1 && apk add --no-cache py3-pip) >/dev/null 2>&1
        if command -v pip3 >/dev/null 2>&1; then
            pip3 install --break-system-packages --no-cache-dir pytest >/dev/null 2>&1 \
              || pip3 install --no-cache-dir pytest >/dev/null 2>&1
        fi
    fi
    # 3) Last resort: bootstrap pip via get-pip.py (pure-python, distro-agnostic).
    if ! python3 -c 'import pytest' >/dev/null 2>&1; then
        if command -v curl >/dev/null 2>&1; then
            curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py 2>/dev/null \
              && python3 /tmp/get-pip.py --break-system-packages >/dev/null 2>&1
        elif command -v wget >/dev/null 2>&1; then
            wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py 2>/dev/null \
              && python3 /tmp/get-pip.py --break-system-packages >/dev/null 2>&1
        fi
        if command -v pip3 >/dev/null 2>&1; then
            pip3 install --break-system-packages --no-cache-dir pytest >/dev/null 2>&1 \
              || pip3 install --no-cache-dir pytest >/dev/null 2>&1
        fi
    fi
    set -e
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_task_yaml(path: Path) -> Dict[str, Any]:
    """Parse a ``task.yaml`` into a dict, returning {} on any failure."""
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("PyYAML not installed; TerminalTraj metadata will be empty")
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, Exception) as exc:  # yaml.YAMLError etc.
        logger.warning("task.yaml parse failed for %s: %s", path, exc)
        return {}


def _inject_pytest_bootstrap(def_text: str) -> str:
    """Insert the pytest-bootstrap block at the top of %post.

    ``_dockerfile_to_apptainer_def`` always produces a %post section (it
    always appends ``mkdir /home/user`` etc.), so we just prepend our block
    after the literal ``%post\\n`` header. Idempotent: skips if the marker
    comment is already present.
    """
    if "TerminalTraj pytest bootstrap" in def_text:
        return def_text
    return def_text.replace("%post\n", "%post\n" + _PYTEST_BOOTSTRAP_POST, 1)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TerminalTrajAdapter(Adapter):
    """``m-a-p/TerminalTraj-5k-instances`` (TerminalBench 1.0 layout, tarball)."""

    name = "terminaltraj"
    hf_repo_id = HF_REPO_ID
    default_dst = "rl_data/output/tasks_terminaltraj"

    # -- Fetch -------------------------------------------------------------
    def fetch(self, cache_dir: Path, *, revision: Optional[str] = None) -> Path:
        """Download + extract ``5k_instances.tar.gz``; return the extracted
        task-dir root.

        Re-runs are cheap: ``hf_hub_download`` no-ops on existing files and
        we skip extraction if the output dir already exists. Set
        ``revision`` to pin a specific dataset commit.
        """
        from huggingface_hub import hf_hub_download

        cache_dir.mkdir(parents=True, exist_ok=True)
        tar_path = Path(hf_hub_download(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            filename=TARBALL_NAME,
            revision=revision,
            local_dir=str(cache_dir / "hf"),
        ))

        extract_root = cache_dir / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        extracted_tasks = extract_root / EXTRACTED_ROOT_NAME
        if not extracted_tasks.is_dir():
            logger.info("Extracting %s into %s ...", tar_path, extract_root)
            with tarfile.open(tar_path, "r:gz") as tf:
                # PEP 706: explicit ``filter="data"`` silences the 3.14
                # DeprecationWarning and strips unsafe metadata. The tarball
                # contains ordinary task files and no device/symlink tricks,
                # so ``data`` is the right choice here.
                try:
                    tf.extractall(extract_root, filter="data")
                except TypeError:
                    # Python < 3.12 doesn't accept ``filter``; fall back.
                    tf.extractall(extract_root)

        if not extracted_tasks.is_dir():
            raise RuntimeError(
                f"Extraction of {tar_path} did not produce {extracted_tasks}"
            )
        return extracted_tasks

    # -- Iterate source task dirs ------------------------------------------
    # The default ``list_source_tasks`` walks direct children, which is
    # exactly what we want (tarball layout is ``5k_instances/task_<N>/``).

    # -- Convert one task --------------------------------------------------
    def convert_one(self, src: Path, dst_root: Path) -> Optional[str]:
        dockerfile = src / "Dockerfile"
        task_yaml = src / "task.yaml"
        test_outputs = src / "tests" / "test_outputs.py"
        # A valid TerminalTraj task ships all three; anything missing is
        # either a stray directory in the tarball (e.g. a chunk/shard file)
        # or a corrupted upstream task — skip it.
        if not (dockerfile.exists() and task_yaml.exists() and test_outputs.exists()):
            return None

        yaml_data = _load_task_yaml(task_yaml)
        instruction = str(yaml_data.get("instruction") or "").strip()

        # item_info.json carries upstream repo provenance (GitHub URL of the
        # source repo, the original Dockerfile path, etc.). Preserve in
        # task.json so downstream analysis can cite sources.
        item_info: Dict[str, Any] = {}
        item_path = src / "item_info.json"
        if item_path.exists():
            try:
                item_info = json.loads(item_path.read_text())
            except (OSError, json.JSONDecodeError):
                item_info = {}

        # Task names: upstream slugs are ``task_<N>``; prefix with ``tt_``
        # so they don't collide with OT-Agent-v1-RL's ``task_<N>``.
        task_name = "tt_" + re.sub(r"\s+", "_", src.name)
        out = dst_root / task_name
        out.mkdir(parents=True, exist_ok=True)

        tags = yaml_data.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        enriched: Dict[str, Any] = {
            "name": task_name,
            # Leave our native taxonomy empty; downstream classifier fills
            # classified_* fields. The upstream category/difficulty/tags
            # are placeholder constants ("mathematics"/"easy") for every
            # task in this release, so they can't substitute for the
            # classifier output.
            "domain": "unknown",
            "skill_type": "unknown",
            "primitive_skills": [],
            "task_complexity": "unknown",
            "command_complexity": "unknown",
            "scenario": "",
            "language": "any (model's choice)",
            "description": instruction,
            "truth": "",
            # Dataset-native metadata preserved for the appendix panels.
            "tt_category": yaml_data.get("category", ""),
            "tt_difficulty": yaml_data.get("difficulty", ""),
            "tt_tags": tags,
            "tt_author_name": yaml_data.get("author_name", ""),
            "tt_parser_name": yaml_data.get("parser_name", ""),
            "tt_max_agent_timeout_sec": yaml_data.get("max_agent_timeout_sec"),
            "tt_max_test_timeout_sec": yaml_data.get("max_test_timeout_sec"),
            "tt_expert_time_estimate_min": yaml_data.get("expert_time_estimate_min"),
            "tt_junior_time_estimate_min": yaml_data.get("junior_time_estimate_min"),
            # Upstream repo provenance from item_info.json.
            "tt_upstream_repo": item_info.get("repo", ""),
            "tt_upstream_dockerfile": item_info.get("dockerfile", ""),
            "tt_docker_image": item_info.get("new_tag", ""),
            # Provenance.
            "source": "tt",
            "source_repo": HF_REPO_ID,
            "source_slug": src.name,
        }
        (out / "task.json").write_text(json.dumps(enriched, indent=2))

        # Derive container.def from the single-line Dockerfile. No COPY
        # lines to resolve, but pass build_context_dir defensively so any
        # task that happens to ship extra files (none observed in this
        # release) doesn't silently drop them.
        derived = _dockerfile_to_apptainer_def(
            dockerfile.read_text(),
            build_context_dir=out,
        )
        # Inject the pytest bootstrap into %post so the SIF ships with a
        # working `pytest` on PATH regardless of which distro the FROM
        # layer used.
        derived = _inject_pytest_bootstrap(derived)
        (out / "container.def").write_text(derived)

        shutil.copy2(test_outputs, out / "test_final_state.py")
        (out / "test_initial_state.py").write_text(_PLACEHOLDER_INITIAL_STATE)

        return task_name


register_adapter(TerminalTrajAdapter())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_terminaltraj_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(TerminalTrajAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true",
                    help="Reuse the already-extracted tarball in cache-dir "
                         "instead of re-downloading from the HF Hub.")
    ap.add_argument("--revision", type=str, default=None,
                    help="Dataset revision (commit SHA or tag) to pin.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = TerminalTrajAdapter()

    if args.skip_download:
        snapshot = (args.cache_dir / "extracted" / EXTRACTED_ROOT_NAME).resolve()
        logger.info("Skipping download; using %s", snapshot)
        if not snapshot.exists():
            logger.error(
                "Extracted tarball at %s does not exist yet; cannot "
                "--skip-download on a cold cache. Run once without the "
                "flag first.", snapshot,
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
