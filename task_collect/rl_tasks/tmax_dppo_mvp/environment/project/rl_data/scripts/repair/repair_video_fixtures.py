"""Re-materialise v2 video fixtures skipped when ffmpeg was missing on the host.

Video bytes are produced **on the task-generation host** (see ``fixture_gen``),
not inside ``base_intricate.sif``. Updating the SIF alone does not fix broken
``video.unavailable.txt`` placeholders.

This script finds ``task_*`` dirs with ``fixture_kind == "video"`` and
``fixtures/video.unavailable.txt``, generates ``video.mp4`` + ``video.label.txt``
using the same deterministic seed as :func:`fixture_seed_for_task`, and rewrites
the ``%files`` block in ``container.def``.

Prerequisites
-------------
* A working ``ffmpeg`` — either on ``PATH``, or set ``FFMPEG_BINARY`` (e.g.
  ``/usr/bin/ffmpeg`` inside ``base_intricate.sif``).
* Login nodes often lack ffmpeg; use ``run_repair_video_fixtures_in_sif.sh``
  (runs this module via ``uv`` on the host and invokes ``ffmpeg`` inside
  ``base_intricate.sif`` through a tiny wrapper).

After running, rebuild per-task ``container.sif`` (or your batch image build)
so Apptainer picks up the new ``%files`` sources.

Examples
--------
.. code-block:: bash

    # Host has ffmpeg
    uv run python -m rl_data.scripts.repair.repair_video_fixtures \\
        --corpus-dir rl_data/output/tasks_skill_tax_v2_20260506_5k

    # Login node: use base_intricate.sif (see run_repair_video_fixtures_in_sif.sh)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rl_data.generator.container_def_patch import replace_apptainer_files_section
from rl_data.generator.fixture_gen import (
    emit_files_section,
    fixture_seed_for_task,
    materialize as fixture_materialize,
    resolve_ffmpeg_binary,
)


_TASK_DIR_RE = re.compile(r"^task_(\d+)_")


def _task_index(task_dir_name: str) -> int:
    m = _TASK_DIR_RE.match(task_dir_name)
    if not m:
        raise ValueError(f"not a task dir name: {task_dir_name!r}")
    return int(m.group(1))


def repair_task(task_dir: Path, *, dry_run: bool) -> bool:
    task_json = task_dir / "task.json"
    if not task_json.is_file():
        return False
    meta = json.loads(task_json.read_text())
    if meta.get("fixture_kind") != "video":
        return False
    sentinel = task_dir / "fixtures" / "video.unavailable.txt"
    if not sentinel.is_file():
        return False

    idx = _task_index(task_dir.name)
    desc = meta.get("description", "")
    truth = meta.get("truth", "")
    seed = fixture_seed_for_task(idx, task_dir.name)

    if dry_run:
        print(f"  would repair: {task_dir.name}")
        return True

    sentinel.unlink(missing_ok=True)
    pairs = fixture_materialize(
        "video",
        task_description=desc,
        truth=truth,
        dest_dir=task_dir,
        seed=seed,
    )
    if not pairs or any("unavailable" in str(h) for h, _ in pairs):
        print(f"  [fail] {task_dir.name}: fixture_materialize did not produce mp4", file=sys.stderr)
        return False

    files_section = emit_files_section(pairs)
    def_path = task_dir / "container.def"
    if not def_path.is_file():
        print(f"  [warn] {task_dir.name}: no container.def; fixtures updated only", file=sys.stderr)
        return True

    old = def_path.read_text()
    def_path.write_text(replace_apptainer_files_section(old, files_section))
    print(f"  ok: {task_dir.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Root containing task_* directories (e.g. v2 5k output).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List tasks that would be repaired; do not write.",
    )
    args = ap.parse_args()
    root = args.corpus_dir.resolve()
    if not root.is_dir():
        print(f"[error] not a directory: {root}", file=sys.stderr)
        return 2

    if not args.dry_run and resolve_ffmpeg_binary() is None:
        print(
            "[error] ffmpeg not found (PATH empty and FFMPEG_BINARY unset/invalid).\n"
            "  Option A: export FFMPEG_BINARY=/path/to/ffmpeg\n"
            "  Option B: rl_data/scripts/repair/run_repair_video_fixtures_in_sif.sh",
            file=sys.stderr,
        )
        return 3

    candidates = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    )
    n = 0
    for d in candidates:
        if repair_task(d, dry_run=args.dry_run):
            n += 1

    print(f"\n{'Would repair' if args.dry_run else 'Repaired'} {n} video task(s).")
    if not args.dry_run and n:
        print(
            "\nNext: rebuild container.sif for affected tasks (or your stage-4 batch) "
            "so %files changes take effect.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
