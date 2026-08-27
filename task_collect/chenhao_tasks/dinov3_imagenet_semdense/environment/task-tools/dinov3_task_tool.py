#!/usr/bin/env python3
"""Frozen task tool for the dinov3-imagenet-semdense task.

Subcommands:
  summarize --attempt <dir>   real-evidence summary of one attempt
  stage --checkpoint <pt> --model-config <config name>
        [--backbone-prefix teacher.backbone.]
                              stage exactly one submission: extracts the
                              tensor-only BACKBONE (SSL heads and student
                              copies are excluded from submission)
  audit --output-root <dir>   pre-verifier self-audit of required outputs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return "(missing)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def cmd_summarize(args: argparse.Namespace) -> int:
    attempt = Path(args.attempt)
    status_path = attempt / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    print(json.dumps({
        "attempt": attempt.name,
        "state": status.get("state", "unknown"),
        "charged_gpu_hours": status.get("charged_gpu_hours"),
        "gpu_hours_cap": status.get("gpu_hours_cap"),
        "run_kind": status.get("run_kind"),
        "start_utc": status.get("start_utc"),
        "end_utc": status.get("end_utc"),
        "checkpoints": sorted(str(p.relative_to(attempt))
                              for p in attempt.rglob("*.pth"))
                       + sorted(str(p.relative_to(attempt))
                                for p in attempt.rglob("*.pt")),
    }, indent=2))
    print("--- last training log lines ---")
    print(tail(attempt / "logs" / "train.log"))
    return 0


def extract_backbone(payload, prefix: str) -> dict:
    """Pull the backbone subtree out of a trainer checkpoint.

    DINOv3 SSL checkpoints hold student/teacher trees; the submission is the
    tensor-only backbone (default: the teacher backbone, the standard
    inference branch). Falls back to the flat dict when it already looks
    like a bare backbone state dict.
    """
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "teacher"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    if not isinstance(state, dict):
        return {}
    prefixed = {k[len(prefix):]: v for k, v in state.items()
                if k.startswith(prefix) and hasattr(v, "dtype")}
    if prefixed:
        return prefixed
    return {k: v for k, v in state.items() if hasattr(v, "dtype")}


def cmd_stage(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    import torch  # local import: heavy

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors = extract_backbone(payload, args.backbone_prefix)
    if not tensors:
        print("error: could not extract a backbone state dict "
              f"(prefix {args.backbone_prefix!r})", file=sys.stderr)
        return 1
    parameters = sum(int(v.numel()) for v in tensors.values())

    submission = output_root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, submission / "backbone.pt")
    torch.save(tensors, output_root / "model_state.pt")

    selection = {
        "source_checkpoint": str(checkpoint),
        "model_config": args.model_config,
        "backbone_prefix": args.backbone_prefix,
        "parameters": parameters,
        "staged": {
            "checkpoint": str(submission / "backbone.pt"),
            "model_state": str(output_root / "model_state.pt"),
        },
    }
    (output_root / "submission-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))
    print("Staged. Now write provenance.json (model_config must equal the "
          "staged name) and run: dinov3_task_tool.py audit")
    return 0


REQUIRED_PROVENANCE = [
    "model_config", "parameters", "training_data", "images_seen",
    "gpu_hours", "label_usage", "training_command", "upstream_commits",
    "pretrained_weights", "evaluation_commands", "optimization_attempts",
    "web_search",
]


def cmd_audit(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    problems: list[str] = []

    selection_path = output_root / "submission-selection.json"
    checkpoint = output_root / "submission" / "backbone.pt"
    model_state = output_root / "model_state.pt"
    for path in (selection_path, checkpoint, model_state):
        if not path.exists():
            problems.append(f"missing required artifact: {path}")

    provenance_path = output_root / "provenance.json"
    provenance = {}
    if not provenance_path.exists():
        problems.append("missing provenance.json")
    else:
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"provenance.json is not valid JSON: {exc}")
        for field in REQUIRED_PROVENANCE:
            if provenance.get(field) in (None, "", []):
                problems.append(f"provenance missing field: {field}")
        if provenance.get("pretrained_weights") not in ("none", "None"):
            problems.append("provenance.pretrained_weights must be 'none'")
        if provenance.get("label_usage") != "none_during_training":
            problems.append("provenance.label_usage must be 'none_during_training'")
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if provenance.get("model_config") != selection.get("model_config"):
                problems.append("provenance.model_config != staged model_config")

    ledger_path = output_root / "experiments.jsonl"
    selected_rows = 0
    ledger_gpu_hours = 0.0
    if not ledger_path.exists():
        problems.append("missing experiments.jsonl")
    else:
        for lineno, line in enumerate(
                ledger_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"experiments.jsonl line {lineno}: invalid JSON")
                continue
            if row.get("status") == "selected":
                selected_rows += 1
            if row.get("label_usage") not in (None, "none_during_training"):
                problems.append(
                    f"experiments.jsonl line {lineno}: label_usage must be 'none_during_training'")
            try:
                ledger_gpu_hours += float(row.get("gpu_hours", 0.0))
            except (TypeError, ValueError):
                problems.append(f"experiments.jsonl line {lineno}: bad gpu_hours")
        if selected_rows != 1:
            problems.append(
                f"experiments.jsonl must contain exactly one 'selected' row, found {selected_rows}")

    measured_gpu_hours = 0.0
    attempts_dir = output_root / "attempts"
    if attempts_dir.exists():
        for status_path in attempts_dir.glob("*/status.json"):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                measured_gpu_hours += float(status.get("charged_gpu_hours", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                problems.append(f"unreadable status: {status_path}")
    if ledger_path.exists() and attempts_dir.exists():
        if measured_gpu_hours > 0 and abs(ledger_gpu_hours - measured_gpu_hours) > \
                max(0.5, 0.05 * measured_gpu_hours):
            problems.append(
                "ledger GPU-hours ({:.2f}) diverge from measured status.json GPU-hours "
                "({:.2f}); reconcile before submitting".format(
                    ledger_gpu_hours, measured_gpu_hours))

    report = {
        "ok": not problems,
        "problems": problems,
        "ledger_gpu_hours": round(ledger_gpu_hours, 3),
        "measured_gpu_hours": round(measured_gpu_hours, 3),
    }
    (output_root / "audit-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_summarize = sub.add_parser("summarize")
    p_summarize.add_argument("--attempt", required=True)

    p_stage = sub.add_parser("stage")
    p_stage.add_argument("--checkpoint", required=True)
    p_stage.add_argument("--model-config", required=True)
    p_stage.add_argument("--backbone-prefix", default="teacher.backbone.")
    p_stage.add_argument("--output-root", default="/app/output")

    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--output-root", default="/app/output")

    args = parser.parse_args()
    if args.cmd == "summarize":
        return cmd_summarize(args)
    if args.cmd == "stage":
        return cmd_stage(args)
    if args.cmd == "audit":
        return cmd_audit(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
