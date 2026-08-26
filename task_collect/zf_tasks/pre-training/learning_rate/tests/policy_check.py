#!/usr/bin/env python3
"""Deterministic source-scope gate for the learning-rate task."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


LR_CANDIDATE = "examples/training/lr_schedule/runtime/lr_schedule_candidate.py"
ALLOWED_IMPORT_ROOTS = {"__future__", "math"}
FORBIDDEN_CALLS = {
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
MUTATING_METHODS = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
}


def persistent_target(target: ast.expr) -> bool:
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        return True
    if isinstance(target, (ast.List, ast.Tuple)):
        return any(persistent_target(item) for item in target.elts)
    return False


def schedule_semantic_violations(path: Path) -> list[str]:
    """Reject metadata, I/O, dynamic-code, and legacy controller surfaces."""

    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        return [f"candidate is not parseable: {error}"]
    violations: list[str] = []
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "build_schedule" not in functions:
        violations.append("candidate must define build_schedule")
    if "build_controller" in functions:
        violations.append("legacy build_controller surface is forbidden")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    violations.append(
                        f"line {node.lineno}: import {alias.name} is forbidden; only math is allowed"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                violations.append(
                    f"line {node.lineno}: import from {module} is forbidden; only math is allowed"
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                violations.append(
                    f"line {node.lineno}: dynamic or I/O call {node.func.id} is forbidden"
                )
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(
                f"line {node.lineno}: persistent mutation via global/nonlocal is forbidden"
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(persistent_target(target) for target in targets):
                violations.append(
                    f"line {node.lineno}: persistent mutation of attributes or containers is forbidden"
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MUTATING_METHODS
        ):
            violations.append(
                f"line {node.lineno}: persistent mutation method {node.func.attr} is forbidden"
            )
    return violations


def ignored(relative: str) -> bool:
    parts = Path(relative).parts
    generated_helper = bool(
        Path(relative).parent.as_posix()
        == "3rdparty/Megatron-LM/megatron/core/datasets"
        and Path(relative).name.startswith("helpers_cpp.")
        and Path(relative).suffix == ".so"
    )
    return (
        ".git" in parts
        or "__pycache__" in parts
        or relative.endswith(".pyc")
        or generated_helper
    )


def inventory(root: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if ignored(relative) or path.is_dir():
            continue
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            result[relative] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            result[relative] = ("other", "")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    policy: dict[str, Any] = yaml.safe_load(args.policy.read_text())
    allowed = tuple(str(item) for item in policy["allowed_source_globs"])
    clean = inventory(args.clean)
    candidate = inventory(args.candidate)
    changed = sorted(
        path
        for path in clean.keys() | candidate.keys()
        if clean.get(path) != candidate.get(path)
    )
    violations = [
        path
        for path in changed
        if not any(fnmatch.fnmatchcase(path, pattern) for pattern in allowed)
    ]
    semantic_violations = schedule_semantic_violations(args.candidate / LR_CANDIDATE)
    report = {
        "policy_gate": 0 if violations or semantic_violations else 1,
        "changed_paths": changed,
        "allowed_source_globs": list(allowed),
        "violations": violations,
        "semantic_violations": semantic_violations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if violations or semantic_violations:
        messages = []
        if violations:
            messages.append("source-scope violations: " + ", ".join(violations))
        if semantic_violations:
            messages.append(
                "schedule semantic violations: " + "; ".join(semantic_violations)
            )
        raise SystemExit(" | ".join(messages))


if __name__ == "__main__":
    main()
