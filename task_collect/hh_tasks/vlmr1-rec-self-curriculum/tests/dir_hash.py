#!/usr/bin/env python3
"""Order-stable directory hash. ONE implementation, two callers.

tests/evaluate.py imports this for the CK-1 base-weight identity check, and
environment/fetch_assets.sh runs it as a script to record the staged model's hash into the
lockfile. Those two numbers MUST be the same function of the same bytes: if the fetcher wrote
one hash and the evaluator computed another, CK-1 would fail on every round for a reason that
looks like tampering. A second implementation is how that happens, so there is only one.

Stdlib only, deliberately -- the fetcher runs on a staging host where torch is not installed.

    python tests/dir_hash.py /path/to/dir
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def dir_manifest_sha256(root: Path) -> str:
    """sha256 over (relative path, size, content hash) of every file, in sorted path order.

    Path and size are folded in so a rename or a truncation changes the digest even when the
    content bytes are a permutation of what was there before.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
        inner = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                inner.update(chunk)
        digest.update(inner.digest())
    return digest.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: dir_hash.py <dir>", file=sys.stderr)
        raise SystemExit(2)
    target = Path(sys.argv[1])
    if not target.is_dir():
        print(f"not a directory: {target}", file=sys.stderr)
        raise SystemExit(2)
    print(dir_manifest_sha256(target))
