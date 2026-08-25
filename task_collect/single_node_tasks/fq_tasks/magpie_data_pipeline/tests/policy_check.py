#!/usr/bin/env python3
"""Run the Magpie hard-policy audit without writing any reward files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policy_contract import audit_candidate
from task_contract import load_llama3_chat_template


def _tokenizer_and_cap():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/models/Meta-Llama-3-8B", local_files_only=True)
    budget = json.loads(Path("/task-assets/baseline_budget.json").read_text(encoding="utf-8"))
    cap = budget.get("canonical_tokens") if isinstance(budget, dict) else None
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("baseline_budget.json must contain canonical_tokens")
    return tokenizer, cap


def _hidden_prompts():
    manifest = json.loads(Path("/task-assets/eval_manifest.json").read_text(encoding="utf-8"))
    values = []
    for entries in manifest.get("sources", {}).values():
        for entry in entries:
            if "prompt" in entry:
                values.append(entry["prompt"])
            else:
                values.extend(message["content"] for message in entry.get("messages", []) if message.get("role") == "user")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean", type=Path, default=Path("/opt/project"))
    parser.add_argument("--report", type=Path, default=Path("/app/output/verifier/policy.json"))
    args = parser.parse_args()
    tokenizer, cap = _tokenizer_and_cap()
    errors = audit_candidate(
        args.clean,
        args.project,
        args.output,
        tokenizer=tokenizer,
        baseline_token_cap=cap,
        hidden_prompts=_hidden_prompts(),
        chat_template=load_llama3_chat_template(),
    )
    payload = {"decision": "PASS" if not errors else "FAIL", "errors": errors}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
