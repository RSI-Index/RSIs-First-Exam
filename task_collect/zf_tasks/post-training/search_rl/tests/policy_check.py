#!/usr/bin/env python3
"""Verifier-private policy and source-isolation checks for search-rl."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_RULES = ("SEARCH-RL-001", "SEARCH-RL-002", "SEARCH-RL-003", "SEARCH-RL-004")
REQUIRED_LAUNCHER = Path(
    "training_scripts/rl/recipe/deepresearch/run_deepresearch_fully_async_megatron.sh"
)


def check(project: Path, policy: Path | None, task: str) -> None:
    project = project.resolve(strict=True)
    if task != "search-rl":
        raise ValueError("unexpected task identity")
    if (project / "evaluation").exists() or (project / ".git").exists():
        raise ValueError("candidate source contains a forbidden evaluation or VCS tree")
    if not (project / REQUIRED_LAUNCHER).is_file():
        raise ValueError("pinned QUEST RL launcher is absent")
    if policy is None:
        return
    text = policy.read_text(encoding="utf-8")
    if "task: search-rl" not in text or '  - "**"' not in text:
        raise ValueError("policy task or editable source scope is invalid")
    for rule in REQUIRED_RULES:
        if f"id: {rule}" not in text:
            raise ValueError(f"policy is missing {rule}")
    for protected in ("/tests/**", "/solution/**", "/task-tools/**", "/logs/verifier/**"):
        if protected not in text:
            raise ValueError(f"policy does not protect {protected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    try:
        check(args.project, args.policy, args.task)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
