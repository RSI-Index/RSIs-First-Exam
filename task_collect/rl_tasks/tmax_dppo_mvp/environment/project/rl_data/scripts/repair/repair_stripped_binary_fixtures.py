"""Re-materialise v2 stripped_binary fixtures that failed (missing gcc or bad C template).

Historically ``rolling_sum`` used ``%%`` in the emitted ``.c`` source (invalid C).
That is fixed in ``fixture_gen``; this script rebuilds ``oracle`` + sidecars and
rewrites ``container.def`` ``%files`` for tasks that still have
``fixtures/binary.unavailable.txt``.

Use ``run_repair_stripped_binary_in_sif.sh`` on login nodes without ``gcc``.
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
    resolve_gcc_binary,
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
    if meta.get("fixture_kind") != "stripped_binary":
        return False
    sentinel = task_dir / "fixtures" / "binary.unavailable.txt"
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
    for stale in ("oracle", "binary.algo.txt"):
        p = task_dir / "fixtures" / stale
        if p.exists():
            p.unlink()

    pairs = fixture_materialize(
        "stripped_binary",
        task_description=desc,
        truth=truth,
        dest_dir=task_dir,
        seed=seed,
    )
    if not pairs or any("unavailable" in str(h) for h, _ in pairs):
        print(
            f"  [fail] {task_dir.name}: fixture_materialize did not produce oracle",
            file=sys.stderr,
        )
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
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.corpus_dir.resolve()
    if not root.is_dir():
        print(f"[error] not a directory: {root}", file=sys.stderr)
        return 2

    if not args.dry_run and resolve_gcc_binary() is None:
        print(
            "[error] gcc not found (PATH empty and GCC_BINARY unset/invalid).\n"
            "  Option A: export GCC_BINARY=/path/to/gcc\n"
            "  Option B: rl_data/scripts/repair/run_repair_stripped_binary_in_sif.sh",
            file=sys.stderr,
        )
        return 3

    candidates = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("task_"))
    n = sum(1 for d in candidates if repair_task(d, dry_run=args.dry_run))

    print(f"\n{'Would repair' if args.dry_run else 'Repaired'} {n} stripped_binary task(s).")
    if not args.dry_run and n:
        print(
            "\nNext: rebuild container.sif for affected tasks (or your stage-4 batch) "
            "so %files changes take effect.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
