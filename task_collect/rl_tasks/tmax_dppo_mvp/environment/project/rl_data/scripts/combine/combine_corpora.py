"""Combine two task corpora into a single "combined" folder of symlinks.

Modes
-----
``balanced`` (default)
    SFT-oriented mix: keep all non-intricate tasks from both corpora, then
    randomly down-sample v2 intricate tasks to hit ``--total`` (see module
    body for the full rule).

``union``
    RL-oriented mix: symlink **every** task from both trees (legacy + v2).
    Fails fast if a ``task_*`` basename appears in both (would clobber a
    symlink). ``--total`` is ignored. Tasks are ordered lexicographically by
    directory name for stable manifests.

Both modes materialise symlinks under ``--out-dir`` so ``rl_data.analyze`` and
upload scripts can use the folder as a drop-in root.

The script refuses to overwrite an existing non-empty ``--out-dir`` unless
``--force`` is passed.

Examples
--------
.. code-block:: bash

    # Balanced ~2.5k SFT view (legacy 1k + v2 2k, intricate capped)
    uv run python -m rl_data.scripts.combine.combine_corpora \\
        --v2-dir rl_data/output/tasks_skill_tax_v2_20260505_2k \\
        --legacy-dir rl_data/output/tasks_skill_tax_20260505_1k_legacy \\
        --out-dir rl_data/output/tasks_skill_tax_combined_20260506_2.5k \\
        --total 2500 \\
        --seed 0

    # Full RL union: legacy ~10k + v2 5k (actual total ≈ task counts on disk)
    uv run python -m rl_data.scripts.combine.combine_corpora \\
        --mode union \\
        --legacy-dir rl_data/output/tasks_skill_tax_20260401_10k \\
        --v2-dir rl_data/output/tasks_skill_tax_v2_20260506_5k \\
        --out-dir rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


_LEGACY_COMPLEXITY_KEYS = ("short", "moderate", "complex")
_INTRICATE_KEY = "intricate"
_ALL_KEYS = _LEGACY_COMPLEXITY_KEYS + (_INTRICATE_KEY,)


def _short_complexity(raw: str) -> str:
    """Map a verbose ``task_complexity`` string back to its short label.

    The full strings live in ``rl_data.generator.task_template_gen.TASK_COMPLEXITY``;
    they all start with ``"<label> task ..."`` (or ``"intricate task ..."``).
    """
    if raw.startswith("intricate"):
        return _INTRICATE_KEY
    return raw.split(" ", 1)[0]


def _load_tasks(root: Path) -> list[tuple[Path, dict]]:
    """Yield ``(task_dir, metadata)`` pairs for every task under ``root``.

    A task dir is any subdir of ``root`` that contains a ``task.json`` file.
    """
    rows: list[tuple[Path, dict]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("task_"):
            continue
        f = d / "task.json"
        if not f.exists():
            continue
        try:
            meta = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  [warn] skipping {d.name}: malformed task.json", file=sys.stderr)
            continue
        rows.append((d, meta))
    return rows


def _bucket_counts(rows: Iterable[tuple[Path, dict]]) -> Counter:
    return Counter(_short_complexity(m["task_complexity"]) for _, m in rows)


def _print_dist(label: str, rows: list[tuple[Path, dict]]) -> None:
    n = len(rows)
    cx = _bucket_counts(rows)
    print(f"  {label} (n={n}):")
    for k in _ALL_KEYS:
        v = cx.get(k, 0)
        pct = (v / n) if n else 0.0
        print(f"    {k:>10s}: {v:>5d}  ({pct:6.2%})")


def select_combined(
    v2_rows: list[tuple[Path, dict]],
    legacy_rows: list[tuple[Path, dict]],
    total: int,
    rng: random.Random,
) -> list[tuple[Path, dict, str]]:
    """Return the chosen tasks tagged with their source corpus.

    Selection rule (see module docstring):
      * keep every non-intricate task from both corpora
      * fill the remaining budget (``total - non_intricate_kept``) by
        randomly sampling from v2 intricate tasks
    """
    keep: list[tuple[Path, dict, str]] = []

    for d, m in legacy_rows:
        if _short_complexity(m["task_complexity"]) != _INTRICATE_KEY:
            keep.append((d, m, "legacy"))
    for d, m in v2_rows:
        if _short_complexity(m["task_complexity"]) != _INTRICATE_KEY:
            keep.append((d, m, "v2"))

    v2_intricate = [
        (d, m) for d, m in v2_rows
        if _short_complexity(m["task_complexity"]) == _INTRICATE_KEY
    ]

    intricate_budget = total - len(keep)
    if intricate_budget < 0:
        # More non-intricate than the requested total. Truncate evenly across
        # the 3 legacy buckets to fit (preserve as much balance as possible).
        print(
            f"  [warn] non-intricate total ({len(keep)}) exceeds --total ({total}); "
            f"truncating to {total} keeping bucket balance",
            file=sys.stderr,
        )
        rng.shuffle(keep)
        # Stable per-bucket truncation: keep up to total/3 from each legacy bucket.
        per_bucket = total // len(_LEGACY_COMPLEXITY_KEYS)
        bucket_taken: Counter = Counter()
        truncated: list[tuple[Path, dict, str]] = []
        for d, m, src in keep:
            k = _short_complexity(m["task_complexity"])
            if bucket_taken[k] >= per_bucket:
                continue
            truncated.append((d, m, src))
            bucket_taken[k] += 1
            if len(truncated) >= total:
                break
        return truncated

    if intricate_budget > len(v2_intricate):
        print(
            f"  [warn] requested {intricate_budget} intricate tasks but only "
            f"{len(v2_intricate)} are available; using all of them",
            file=sys.stderr,
        )
        intricate_budget = len(v2_intricate)

    rng.shuffle(v2_intricate)
    for d, m in v2_intricate[:intricate_budget]:
        keep.append((d, m, "v2"))

    return keep


def select_union(
    v2_rows: list[tuple[Path, dict]],
    legacy_rows: list[tuple[Path, dict]],
) -> list[tuple[Path, dict, str]]:
    """Symlink every task from legacy then v2; stable sort by directory name."""
    legacy_names = [d.name for d, _ in legacy_rows]
    v2_names = [d.name for d, _ in v2_rows]
    overlap = set(legacy_names) & set(v2_names)
    if overlap:
        sample = sorted(overlap)[:10]
        more = f" (+{len(overlap) - len(sample)} more)" if len(overlap) > 10 else ""
        print(
            f"[error] {len(overlap)} task_* basename(s) appear in both corpora "
            f"(example: {sample}{more}). Resolve collisions before union.",
            file=sys.stderr,
        )
        sys.exit(3)

    chosen: list[tuple[Path, dict, str]] = [
        (d, m, "legacy") for d, m in legacy_rows
    ]
    chosen.extend((d, m, "v2") for d, m in v2_rows)
    chosen.sort(key=lambda t: t[0].name)
    return chosen


def materialise(
    chosen: list[tuple[Path, dict, str]],
    out_dir: Path,
    *,
    force: bool,
    combine_mode: str | None = None,
) -> None:
    """Symlink every chosen task dir into ``out_dir``.

    Also writes a ``_combine_manifest.json`` summarising what was included.
    """
    out_dir = out_dir.resolve()
    if out_dir.exists():
        has_content = any(p.name.startswith("task_") for p in out_dir.iterdir())
        if has_content and not force:
            print(
                f"[error] {out_dir} already contains task_* entries; pass --force to "
                f"replace.",
                file=sys.stderr,
            )
            sys.exit(2)
        if has_content and force:
            for p in out_dir.iterdir():
                if p.name.startswith("task_") and p.is_symlink():
                    p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    name_to_source: dict[str, str] = {}
    for src_dir, _meta, src_label in chosen:
        link = out_dir / src_dir.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(src_dir.resolve(), link)
        name_to_source[src_dir.name] = src_label

    manifest = {
        "total": len(chosen),
        "by_source": dict(Counter(s for _, _, s in chosen)),
        "by_complexity": dict(_bucket_counts([(d, m) for d, m, _ in chosen])),
        "by_source_and_complexity": {
            src: dict(_bucket_counts([(d, m) for d, m, s in chosen if s == src]))
            for src in sorted({s for _, _, s in chosen})
        },
        "tasks": name_to_source,
    }
    if combine_mode is not None:
        manifest["combine_mode"] = combine_mode
    (out_dir / "_combine_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"  wrote manifest: {out_dir / '_combine_manifest.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-dir", type=Path, required=True, help="v2 corpus root.")
    p.add_argument("--legacy-dir", type=Path, required=True, help="legacy corpus root.")
    p.add_argument("--out-dir", type=Path, required=True, help="output combined corpus dir.")
    p.add_argument(
        "--mode",
        choices=("balanced", "union"),
        default="balanced",
        help='combination strategy: "balanced" (SFT, uses --total) or '
        '"union" (RL: all tasks from both dirs; ignores --total).',
    )
    p.add_argument(
        "--total", type=int, default=2500,
        help="target total tasks when --mode balanced (ignored for union).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for intricate sampling.")
    p.add_argument(
        "--force", action="store_true",
        help="overwrite existing task_* symlinks in --out-dir if it already exists.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the planned distribution but do not write anything.",
    )
    args = p.parse_args()

    print("[combine] loading source corpora")
    v2_rows = _load_tasks(args.v2_dir)
    legacy_rows = _load_tasks(args.legacy_dir)
    print(f"  v2-dir     = {args.v2_dir}  → {len(v2_rows)} tasks")
    print(f"  legacy-dir = {args.legacy_dir}  → {len(legacy_rows)} tasks")
    print()
    print("[combine] source distributions")
    _print_dist("v2", v2_rows)
    _print_dist("legacy", legacy_rows)

    if args.mode == "union":
        chosen = select_union(v2_rows, legacy_rows)
    else:
        rng = random.Random(args.seed)
        chosen = select_combined(v2_rows, legacy_rows, args.total, rng)

    print()
    print("[combine] selected combined distribution")
    _print_dist("combined", [(d, m) for d, m, _ in chosen])
    by_src = Counter(s for _, _, s in chosen)
    print(f"  by source: {dict(by_src)}")

    if args.dry_run:
        print("\n[combine] --dry-run: not writing.")
        return

    print()
    print(f"[combine] materialising into {args.out_dir}")
    materialise(chosen, args.out_dir, force=args.force, combine_mode=args.mode)
    print(f"  done. {len(chosen)} task symlinks placed.")


if __name__ == "__main__":
    main()
