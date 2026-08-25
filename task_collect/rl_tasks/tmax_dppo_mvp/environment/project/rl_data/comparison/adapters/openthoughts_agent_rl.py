"""Adapter for ``open-thoughts/OpenThoughts-Agent-v1-RL`` (728 tasks).

The dataset ships as a single ``tasks.parquet`` file (~10 MB) with columns:

    path: string        e.g. ``"task_1008"``
    task_binary: bytes  gzipped tarball of the task directory

Each extracted task follows a uniform layout:

    task_NNNN/
      instruction.md            prompt for the model
      task.toml                 metadata (title, difficulty, ...)
      task_manifest.json        original/mutated NL + bash, target output
      environment/
        Dockerfile              always identical across all 728 tasks
        seeds/                  starting files bound into /workspace
      tests/
        test.sh                 always identical (cmp /output/... vs expected)
        expected_output.txt     reference output
      solution/
        solve.sh                reference solution (we don't use)

Because the Dockerfile and test.sh are dataset-wide constants, our per-task
``container.def`` simply copies the per-task payload (seeds + expected_output)
on top of a shared, prebuilt base SIF (``ubuntu_24.04_ot.sif``). The run
script is responsible for building that base SIF once before the pre-build
phase; see ``rl_data/scripts/comparison/run_generate_solutions_openthoughts.sh``.

We do not use ``flatten_harbor_task`` here because the OT-Agent-v1-RL layout
diverges in three non-trivial ways: (1) no ``tests/test_final_state.py``,
(2) a pre-existing (identical) Dockerfile that we want to short-circuit via
``localimage`` bootstrap, and (3) payload that lands in ``/workspace`` via
Docker ``COPY seeds/ /workspace/`` semantics -- directory contents, not the
directory itself.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from rl_data.comparison.adapters import Adapter, register_adapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification shim: our harness runs pytest, not bash. Wrap test.sh in a tiny
# pytest module so nothing in generate_solutions.py has to change.
# ---------------------------------------------------------------------------
_TEST_FINAL_WRAPPER = '''\
"""Pytest wrapper around the dataset\'s tests/test.sh verifier.

OpenThoughts-Agent-v1-RL ships a shell verifier at /tests/test.sh that checks
the task\'s target output file (default /output/command_capture.txt) against a
reference /tests/expected_output.txt. We expose that outcome to our pytest-
based harness so no changes to generate_solutions.py are required.
"""
import subprocess


def test_shell_verifier():
    result = subprocess.run(
        ["bash", "/tests/test.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "shell verifier failed: "
        + (result.stderr.strip() or result.stdout.strip() or "(no output)")
    )
'''


_PLACEHOLDER_INITIAL_STATE = (
    "def test_placeholder_initial_state():\n"
    "    assert True\n"
)


# 92.2% of upstream OT-Agent-v1-RL instructions say:
#   "- From `/workspace`, assemble any missing fixtures using standard shell
#    utilities before running your command."
# The word "assemble" + "missing" strongly biases agents to BLINDLY CREATE
# their own `aa`/`bb` fixtures (with fake contents) instead of inspecting the
# seeds that are already present in the container. This single line destroys
# ~75% of rollouts -- confirmed by inspecting failed trajectories where the
# first turn is `cat > aa <<EOF ... EOF` overwriting the real seed file.
#
# We surgically replace that one line (and a couple of minor variants) with
# an explicit statement that the fixtures are already present and must be
# inspected, not recreated. The rest of the instruction (goal, output path,
# grading criteria) is left verbatim.
_INSTRUCTION_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "- From `/workspace`, assemble any missing fixtures using standard shell utilities before running your command.",
        "- Work from `/workspace`. All required fixture files (e.g. `aa`, `bb`, seed inputs) are **already present** in `/workspace` AND in your current working directory (`/home/user`). **Inspect** them first with `ls`, `cat`, `head`, etc. — DO NOT create new fixture files or overwrite the existing ones.",
    ),
    (
        "From `/workspace`, assemble any missing fixtures using standard shell utilities before running your command.",
        "Work from `/workspace`. All required fixture files are **already present** in `/workspace` and in your current working directory (`/home/user`). Inspect them first with `ls`/`cat`; do NOT create new fixture files.",
    ),
)


def _patch_instruction(text: str) -> str:
    """Apply dataset-wide instruction rewrites (see _INSTRUCTION_REWRITES)."""
    for old, new in _INSTRUCTION_REWRITES:
        text = text.replace(old, new)
    return text


def _build_container_def(
    *,
    task_dir: Path,
    base_sif_relpath: str = "./ubuntu_24.04_ot.sif",
) -> str:
    """Produce a container.def that layers per-task payload on the shared base.

    Payload is addressed by absolute path (rooted at ``task_dir``) so that
    ``apptainer build`` can resolve it regardless of invocation CWD.

    Seeds are copied into **both** ``/workspace`` (matching the upstream
    Dockerfile's ``COPY seeds/ /workspace/``, which the instruction text
    references) **and** ``/home/user`` (where the harness drops the agent on
    shell start -- see ``rl_data.generator.env._materialize_writable_home_user``
    and the ``--pwd /home/user`` flag). Without the ``/home/user`` copy,
    the agent lands in an empty writable tmpfs and often fabricates its own
    fake fixtures instead of navigating to ``/workspace``; with the dual
    copy it sees the seeds regardless of how it interprets the instruction.
    """
    seeds_abs = (task_dir / "environment" / "seeds").resolve()
    test_sh_abs = (task_dir / "tests" / "test.sh").resolve()
    expected_abs = (task_dir / "tests" / "expected_output.txt").resolve()

    lines: List[str] = [
        f"Bootstrap: localimage",
        f"From: {base_sif_relpath}",
        "",
        "%post",
        "    set -e",
        "    mkdir -p /workspace /output /logs/verifier /tests /home/user",
        # Spread seeds contents into /workspace AND /home/user so the agent
        # can reach them both via the instruction-referenced path and via its
        # starting cwd. See docstring above for why both are needed.
        "    if [ -d /opt/_ot_stage/seeds ]; then",
        "        cp -a /opt/_ot_stage/seeds/. /workspace/",
        "        cp -a /opt/_ot_stage/seeds/. /home/user/",
        "    fi",
        "    rm -rf /opt/_ot_stage",
        "    chmod 755 /workspace /home/user",
        "",
        "%labels",
        "    Author openthoughts-agent-v1-rl-adapter",
        "    Description \"OpenThoughts-Agent-v1-RL task layer over ubuntu_24.04_ot base (seeds in /workspace AND /home/user)\"",
        "",
        "%files",
        f"    {seeds_abs} /opt/_ot_stage/seeds",
        f"    {test_sh_abs} /tests/test.sh",
        f"    {expected_abs} /tests/expected_output.txt",
        "",
    ]
    return "\n".join(lines) + "\n"


def _load_task_toml(task_toml_path: Path) -> Dict[str, Any]:
    if not task_toml_path.exists():
        return {}
    try:
        return tomllib.loads(task_toml_path.read_text())
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return {}


class OpenThoughtsAgentRLAdapter(Adapter):
    name = "openthoughts_agent_rl"
    hf_repo_id = "open-thoughts/OpenThoughts-Agent-v1-RL"
    default_dst = "rl_data/output/tasks_openthoughts_agent_rl"

    # ---------- Fetch: download + extract ---------------------------------
    def fetch(self, cache_dir: Path, *, revision: Optional[str] = None,
              skip_download: bool = False) -> Path:
        """Download ``tasks.parquet`` into ``cache_dir`` and extract rows.

        Returns the path to the directory holding per-task dirs. Safe to call
        repeatedly; already-extracted task dirs are reused.

        If ``skip_download`` is True we use the parquet file already present
        in the local HF cache instead of contacting the Hub. The validation
        and re-extract logic below still runs, so corrupted local extracts
        self-heal even when the network is offline.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openthoughts_agent_rl adapter requires pyarrow. "
                "Install it in your environment."
            ) from exc

        cache_dir.mkdir(parents=True, exist_ok=True)
        if skip_download:
            # Locate the parquet that a previous fetch downloaded.
            hf_root = cache_dir / "_hf_cache"
            candidates = list(hf_root.rglob("tasks.parquet")) if hf_root.exists() else []
            if not candidates:
                raise RuntimeError(
                    f"--skip-download requested but no tasks.parquet found in "
                    f"{hf_root}. Run once without --skip-download first."
                )
            pq_path = candidates[0]
            logger.info("Skipping download; using cached parquet %s", pq_path)
        else:
            from huggingface_hub import hf_hub_download
            logger.info("Downloading %s/tasks.parquet ...", self.hf_repo_id)
            pq_path = Path(hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="tasks.parquet",
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir / "_hf_cache"),
            ))

        extracted = cache_dir / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(str(pq_path))
        if "path" not in table.column_names or "task_binary" not in table.column_names:
            raise RuntimeError(
                f"{self.hf_repo_id}/tasks.parquet missing expected columns; "
                f"found {list(table.column_names)}"
            )
        paths = table.column("path").to_pylist()
        datas = table.column("task_binary").to_pylist()

        new_count = 0
        for i, (rel_path, data) in enumerate(zip(paths, datas)):
            if not isinstance(rel_path, str):
                logger.warning("row %d: 'path' is not a string, skipping", i)
                continue
            if not isinstance(data, (bytes, bytearray, memoryview)):
                logger.warning("row %d (%s): 'task_binary' is not bytes, skipping",
                               i, rel_path)
                continue

            # Sanitize rel_path against traversal.
            safe = PurePosixPath(rel_path)
            parts = [p for p in safe.parts if p not in ("..", "")]
            rel = Path(*parts) if parts else Path(f"task_{i}")
            dest = (extracted / rel).resolve()
            try:
                dest.relative_to(extracted.resolve())
            except ValueError:
                logger.warning("row %d (%s): unsafe target, skipping", i, rel_path)
                continue

            # Skip if already extracted AND complete. We previously trusted a
            # `.extracted_ok` marker alone, but observed that empty-file seeds
            # (e.g. ``environment/seeds/alpha/one.example.com``, size 0) and
            # other small files can disappear from /gpfs scratch between runs
            # (filesystem maintenance, accidental cleanup, etc.) while the
            # marker survives. To self-heal, validate that EVERY file the tar
            # archive declares is present on disk; if any are missing, wipe and
            # re-extract.
            marker = dest / ".extracted_ok"
            try:
                with tarfile.open(fileobj=io.BytesIO(bytes(data)), mode="r:*") as tf:
                    expected_files = [m.name for m in tf.getmembers() if m.isfile()]
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not read tar header for %s: %s", rel_path, exc)
                continue

            if marker.exists():
                missing = [f for f in expected_files if not (dest / f).exists()]
                if not missing:
                    continue
                logger.warning(
                    "cache corruption detected for %s (%d/%d files missing, "
                    "e.g. %r); re-extracting",
                    rel_path, len(missing), len(expected_files), missing[:3],
                )

            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(fileobj=io.BytesIO(bytes(data)), mode="r:*") as tf:
                    # tarfile filter="data" in py3.12+ is the safe default.
                    try:
                        tf.extractall(dest, filter="data")
                    except TypeError:
                        tf.extractall(dest)
                marker.write_text("ok\n")
                new_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("extraction failed for %s: %s", rel_path, exc)
                shutil.rmtree(dest, ignore_errors=True)

        logger.info(
            "Extracted %d new task(s) (%d already present) into %s",
            new_count, len(paths) - new_count, extracted,
        )
        return extracted

    # ---------- Convert one task ------------------------------------------
    def convert_one(self, src: Path, dst_root: Path) -> Optional[str]:
        env = src / "environment"
        tests = src / "tests"
        test_sh = tests / "test.sh"
        expected = tests / "expected_output.txt"
        instruction_md = src / "instruction.md"

        # Sanity-check the expected layout.
        if not (env.exists() and tests.exists() and test_sh.exists()
                and expected.exists()):
            logger.warning("convert_one: %s missing expected files; skipping", src.name)
            return None

        task_name = "otrl_" + src.name
        out = dst_root / task_name
        out.mkdir(parents=True, exist_ok=True)

        # -- task.json (metadata + provenance) -----------------------------
        description = ""
        if instruction_md.exists():
            try:
                description = _patch_instruction(instruction_md.read_text())
            except OSError:
                pass

        toml_meta = _load_task_toml(src / "task.toml")

        manifest: Dict[str, Any] = {}
        manifest_path = src / "task_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass

        enriched: Dict[str, Any] = {
            "name": task_name,
            "domain": "unknown",
            "skill_type": "unknown",
            "primitive_skills": [],
            "task_complexity": "unknown",
            "command_complexity": "unknown",
            "scenario": "",
            "language": "any (model's choice)",
            "description": description,
            "truth": manifest.get("mutated_bash") or manifest.get("original_bash", ""),
            # Dataset-native metadata for appendix panels.
            "otrl_category": toml_meta.get("category", ""),
            "otrl_difficulty": toml_meta.get("difficulty", ""),
            "otrl_tags": toml_meta.get("tags", []),
            "otrl_author_name": toml_meta.get("author_name", ""),
            "otrl_original_nl": manifest.get("original_nl", ""),
            "otrl_original_bash": manifest.get("original_bash", ""),
            "otrl_mutated_nl": manifest.get("mutated_nl", ""),
            "otrl_target_output_file": manifest.get("target_output_file", ""),
            # Provenance.
            "source": "otrl",
            "source_repo": self.hf_repo_id,
            "source_slug": src.name,
        }
        (out / "task.json").write_text(json.dumps(enriched, indent=2))

        # -- container.def (shared base + per-task %files) -----------------
        # Materialize a stable copy of the payload alongside the def so the
        # absolute paths in %files don't depend on the ephemeral HF cache.
        for sub in ("environment/seeds", "tests/test.sh", "tests/expected_output.txt"):
            s = src / sub
            d = out / sub
            d.parent.mkdir(parents=True, exist_ok=True)
            try:
                if s.is_dir():
                    if d.exists():
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            except OSError as exc:
                logger.warning("convert_one(%s): failed to stage %s: %s",
                               task_name, sub, exc)

        (out / "container.def").write_text(_build_container_def(task_dir=out))

        # -- Pytest wrapper + placeholder initial-state test ---------------
        (out / "test_final_state.py").write_text(_TEST_FINAL_WRAPPER)
        (out / "test_initial_state.py").write_text(_PLACEHOLDER_INITIAL_STATE)

        return task_name


register_adapter(OpenThoughtsAgentRLAdapter())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_otrl_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(OpenThoughtsAgentRLAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--revision", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = OpenThoughtsAgentRLAdapter()

    # Always go through fetch() so the validate-and-self-heal logic runs even
    # when --skip-download is set; fetch() will use the locally-cached parquet
    # and skip the HF Hub download in that case.
    snapshot = adapter.fetch(
        args.cache_dir.resolve(),
        revision=args.revision,
        skip_download=args.skip_download,
    )

    converted, skipped = adapter.convert_all(
        snapshot, args.dst.resolve(),
        limit=args.limit, workers=args.workers,
    )
    logger.info("Done. converted=%d skipped=%d  dst=%s",
                converted, skipped, args.dst)


if __name__ == "__main__":
    main()
