"""Render the stacked-composition figure with each candidate palette.

Two modes:

1. **Preview** (default) — render every registered palette into
   ``rl_data/output/comparison/_palette_previews/preview_<palette>.png`` so
   you can pick one visually.

2. **Finalize** — once you've picked a palette, write it as the canonical
   main-body figure ``rl_data/output/comparison/main/fig6_composition_domain_stacked.png``
   without re-running the whole comparison pipeline.

Examples::

    # 1. compare all palettes
    uv run python -m rl_data.comparison.preview_palettes

    # 2. lock in a choice
    uv run python -m rl_data.comparison.preview_palettes --finalize anthropic_book

    # 3. fine-tune a single knob during iteration
    uv run python -m rl_data.comparison.preview_palettes \\
        --palettes anthropic_book --annotate-min-pct 3.0 --no-normalize
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

from rl_data.comparison.core import DOMAINS_ORDER, DatasetSpec, RunContext
from rl_data.comparison.modules import _render_stacked_composition
from rl_data.comparison.styles import (
    PALETTES,
    StackedCompositionStyle,
    default_stacked_style,
    list_palettes,
)

logger = logging.getLogger(__name__)


# Friendly display names for the datasets currently in the CSV.
_DISPLAY_NAMES = {
    "skill_tax": "Skill-Tax (ours)",
    "endless_terminals": "Endless-Terminals",
    "openthoughts_agent_rl": "OpenThoughts-Agent-v1-RL",
    "termigen": "TermiGen",
    "terminaltraj": "TerminalTraj",
}


def _load_csv(path: Path) -> tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]], List[str]]:
    """Load the composition CSV emitted by the comparison pipeline.

    Returns ``(bucket_counts, bucket_pct, dataset_order)`` with the dataset
    order preserved from the file (skill_tax first, then baselines).
    """
    bucket_counts: Dict[str, Dict[str, int]] = {}
    bucket_pct: Dict[str, Dict[str, float]] = {}
    seen: List[str] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            bucket = row["bucket"]
            if ds not in bucket_counts:
                bucket_counts[ds] = {}
                bucket_pct[ds] = {}
                seen.append(ds)
            bucket_counts[ds][bucket] = int(row["n_tasks"])
            bucket_pct[ds][bucket] = float(row["pct_tasks"])
    return bucket_counts, bucket_pct, seen


def _make_ctx(
    bucket_counts: Dict[str, Dict[str, int]],
    bucket_pct: Dict[str, Dict[str, float]],
    dataset_order: List[str],
    out_dir: Path,
) -> RunContext:
    specs = [
        DatasetSpec(
            name=ds,
            display_name=_DISPLAY_NAMES.get(ds, ds),
            tasks_dir=Path("/dev/null"),
            color="#1f6feb",
            is_reference=(ds == "skill_tax"),
        )
        for ds in dataset_order
    ]
    return RunContext(
        specs=specs,
        records_by_name={s.name: [] for s in specs},
        model_slug="preview",
        main_dir=out_dir,
        appendix_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv", type=Path,
        default=Path("rl_data/output/comparison/main/fig3_composition_domain_ridgeline.csv"),
        help="Source CSV with composition data (default: the one fig3 emits).",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("rl_data/output/comparison/_palette_previews"),
        help="Output directory for preview mode.",
    )
    ap.add_argument(
        "--palettes", nargs="*", default=None,
        help=f"Subset of palettes to render in preview mode (default: all). "
             f"Available: {', '.join(list_palettes())}",
    )
    ap.add_argument(
        "--finalize", metavar="PALETTE", default=None,
        help="Write the canonical main-body figure "
             "rl_data/output/comparison/main/fig6_composition_domain_stacked.png "
             "using this palette and exit.",
    )
    ap.add_argument(
        "--final-out", type=Path,
        default=Path("rl_data/output/comparison/main/fig6_composition_domain_stacked"),
        help="Path-base (no extension) for --finalize. Default matches the "
             "canonical pipeline output so it overwrites in place.",
    )
    ap.add_argument(
        "--no-normalize", action="store_true",
        help="Render bars in absolute task counts (default: normalize to 100%%).",
    )
    ap.add_argument("--annotate-min-pct", type=float, default=4.0)
    ap.add_argument(
        "--title", default=None,
        help="Override the figure title. Default: 'Domain composition' "
             "(or 'Domain composition · palette: <name>' in preview mode).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    normalize = not args.no_normalize

    bucket_counts, bucket_pct, dataset_order = _load_csv(args.csv.resolve())

    if args.finalize is not None:
        if args.finalize not in PALETTES:
            raise SystemExit(
                f"Unknown palette {args.finalize!r}. Available: {', '.join(list_palettes())}"
            )
        out_base = args.final_out.resolve()
        out_base.parent.mkdir(parents=True, exist_ok=True)
        ctx = _make_ctx(bucket_counts, bucket_pct, dataset_order, out_base.parent)
        style = default_stacked_style(
            title=args.title or "Domain composition",
            palette_name=args.finalize,
            normalize=normalize,
            annotate_min_pct=args.annotate_min_pct,
        )
        _render_stacked_composition(
            ctx, DOMAINS_ORDER, bucket_counts, bucket_pct,
            path_base=out_base, style=style,
        )
        logger.info("Wrote final figure %s.png (palette=%s)", out_base, args.finalize)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = _make_ctx(bucket_counts, bucket_pct, dataset_order, args.out.resolve())
    palettes = args.palettes if args.palettes else list_palettes()
    for palette in palettes:
        if palette not in PALETTES:
            logger.warning("Skipping unknown palette %s", palette)
            continue
        style = default_stacked_style(
            title=args.title or f"Domain composition  ·  palette: {palette}",
            palette_name=palette,
            normalize=normalize,
            annotate_min_pct=args.annotate_min_pct,
        )
        path_base = args.out / f"preview_{palette}"
        _render_stacked_composition(
            ctx, DOMAINS_ORDER, bucket_counts, bucket_pct,
            path_base=path_base, style=style,
        )
        logger.info("Wrote %s.png", path_base)

    logger.info("Done. Previews in %s", args.out)


if __name__ == "__main__":
    main()
