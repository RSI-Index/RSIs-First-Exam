#!/usr/bin/env python3
"""Re-pull the ResearchAI screen designs from Stitch into build/screens/.

Talks to the Stitch MCP endpoint over plain JSON-RPC (no MCP client needed):
`list_screens` for the project, then downloads each screen's htmlCode file.

    STITCH_API_KEY=... python3 build/sync_screens.py
    python3 build/build.py

The API key is never stored in this repo — it is read from the environment.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ENDPOINT = "https://stitch.googleapis.com/mcp"
PROJECT_ID = os.environ.get("STITCH_PROJECT_ID", "1164354532195732221")
OUT = Path(__file__).resolve().parent / "screens"

# Screens the site is built from; anything else in the project is ignored.
#
# The Mission Control and Drafting designs below were deleted from the Stitch
# project after they were built; build/screens/ keeps the last pull, which is
# what build.py still uses. A missing screen is reported, never overwritten, so
# re-syncing cannot silently empty a page.
WANTED = {
    "18a02818c5f341548924c8b5d8b2b0af",  # 任务启动页 — Start
    "5efd1b374cf44c27b5571985b983349b",  # Mission Control (no longer in project)
    "594fd847725f44429cde3167158062b1",  # Drafting (no longer in project)
    "768df3ba9c6747c98d663989eb450714",  # Final Delivery
}


def rpc(key: str, method: str, params: dict, timeout: int = 120) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "X-Goog-Api-Key": key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    if "error" in payload:
        raise SystemExit(f"MCP error: {payload['error']}")
    result = payload["result"]
    if result.get("isError"):
        raise SystemExit(f"tool error: {result['content'][0]['text']}")
    return result


def main() -> int:
    key = os.environ.get("STITCH_API_KEY")
    if not key:
        print("set STITCH_API_KEY (X-Goog-Api-Key for stitch.googleapis.com)", file=sys.stderr)
        return 1

    result = rpc(key, "tools/call", {"name": "list_screens", "arguments": {"projectId": PROJECT_ID}})
    screens = json.loads(result["content"][0]["text"])["screens"]
    OUT.mkdir(parents=True, exist_ok=True)

    written = 0
    for screen in screens:
        sid = screen["name"].split("/")[-1]
        if sid not in WANTED:
            continue
        url = (screen.get("htmlCode") or {}).get("downloadUrl")
        if not url:
            print(f"  skip {sid}: no htmlCode", file=sys.stderr)
            continue
        data = urllib.request.urlopen(url, timeout=120).read()
        (OUT / f"{sid}.html").write_bytes(data)
        print(f"  {sid}  {screen.get('title', '')}  ({len(data):,} bytes)")
        written += 1

    missing = WANTED - {s["name"].split("/")[-1] for s in screens}
    if missing:
        print(f"warning: not found in project {PROJECT_ID}: {', '.join(sorted(missing))}", file=sys.stderr)
    print(f"synced {written} screen(s) into {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
