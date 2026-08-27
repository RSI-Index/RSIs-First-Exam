#!/usr/bin/env python3
"""Stage a deterministic root-only hidden Search-R1 evaluation slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EVAL_RECORDS = 512


def select_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quota = {"nq": EVAL_RECORDS // 2, "hotpotqa": EVAL_RECORDS // 2}
    selected = []
    counts = {name: 0 for name in quota}
    for row in rows:
        source = row.get("data_source")
        if source in quota and counts[source] < quota[source]:
            selected.append(row)
            counts[source] += 1
    if counts != quota:
        raise ValueError(
            f"source test split must provide exactly {EVAL_RECORDS} selected rows "
            f"(256 nq and 256 hotpotqa); found {counts}"
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    source_table = pq.read_table(args.source)
    rows = select_eval_rows(source_table.to_pylist())
    output_table = pa.Table.from_pylist(rows, schema=source_table.schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, args.output, compression="zstd")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.manifest.write_text(
        json.dumps(
            {"records": EVAL_RECORDS, "sha256": digest, "source_order": "first_256_each_nq_hotpotqa"},
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
