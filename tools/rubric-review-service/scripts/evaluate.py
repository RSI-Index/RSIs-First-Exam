#!/usr/bin/env python3
"""Send a local task proposal to the private Worker review API."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal", type=Path)
    args = parser.parse_args()

    url = os.environ.get("JUDGE_URL")
    judge_key = os.environ.get("JUDGE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not all((url, judge_key, openai_key)):
        sys.exit("Set JUDGE_URL, JUDGE_API_KEY, and OPENAI_API_KEY.")

    payload = {"proposal": args.proposal.read_text(encoding="utf-8")}
    request = Request(
        url.rstrip("/") + "/v1/reviews/task-proposal",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {judge_key}",
            "X-OpenAI-API-Key": f"Bearer {openai_key}",
            "Content-Type": "application/json",
            "User-Agent": "rsi-index-rubric-review/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            print(response.read().decode("utf-8"))
    except HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
