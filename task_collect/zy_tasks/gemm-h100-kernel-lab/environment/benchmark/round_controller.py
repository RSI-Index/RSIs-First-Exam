#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"gemm_v([0-9]{3})(?:_|\.cu)")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def iopt(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def gemm_root(app_dir: Path) -> Path:
    candidates = [
        app_dir / "workdir" / "gemm_lab",
        app_dir / "environment" / "workdir" / "gemm_lab",
        app_dir,
    ]
    for root in candidates:
        if (root / "CMakeLists.txt").exists() and (root / "kernels").exists():
            return root
    raise FileNotFoundError(f"cannot locate gemm_lab under {app_dir}")


def discover_versions(root: Path) -> list[int]:
    versions: set[int] = set()
    for path in (root / "kernels").glob("gemm_v*.cu"):
        match = VERSION_RE.search(path.name)
        if match:
            versions.add(int(match.group(1)))
    return sorted(versions)


def selected_version(root: Path, versions: list[int]) -> int | None:
    final_file = root / "FINAL_VERSION"
    if final_file.exists():
        match = re.search(r"[0-9]+", final_file.read_text(errors="ignore"))
        if match:
            return int(match.group(0))
    return versions[-1] if versions else None


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def public_env_for_tier(tier: str) -> dict[str, str]:
    if tier == "smoke":
        return {
            "GEMM_PUBLIC_SHAPES": os.environ.get("GEMM_SMOKE_SHAPES", "512x512x512"),
            "GEMM_PUBLIC_WARMUP": os.environ.get("GEMM_SMOKE_WARMUP", "1"),
            "GEMM_PUBLIC_ITERS": os.environ.get("GEMM_SMOKE_ITERS", "2"),
            "GEMM_PUBLIC_REPEATS": os.environ.get("GEMM_SMOKE_REPEATS", "1"),
            "GEMM_PUBLIC_SAMPLES": os.environ.get("GEMM_SMOKE_SAMPLES", "512"),
        }
    return {}


def run_public_feedback(args: argparse.Namespace) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(public_env_for_tier(args.tier))
    env["CUDA_HOME"] = env.get("CUDA_HOME", "/usr/local/cuda")
    env["APP_DIR"] = str(args.app_dir)
    env.setdefault("GEMM_CONTINUATION_ARTIFACTS_DIR", str(args.rounds_dir))
    cmd = ["bash", str(args.feedback_script)]
    proc = subprocess.run(
        cmd,
        cwd=str(args.app_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    return proc.returncode, proc.stdout


def snapshot_public(benchmark_dir: Path, round_dir: Path) -> None:
    public_dir = round_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "public_results.json",
        "public_feedback.md",
        "public_history.jsonl",
        "candidate_eval_latest.json",
        "candidate_eval_history.jsonl",
        "candidate_manifest.jsonl",
        "optimization_state.json",
    ):
        copy_file_if_exists(benchmark_dir / name, public_dir / name)
    copy_tree_if_exists(benchmark_dir / "candidate_evals", public_dir / "candidate_evals")


def snapshot_code(root: Path, round_dir: Path, previous_round_dir: Path | None) -> None:
    code_dir = round_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in ("FINAL_VERSION", "notes.md", "CMakeLists.txt", "README_AGENT.md"):
        copy_file_if_exists(root / name, code_dir / name)
    for name in ("kernels", "include", "src", "tools"):
        copy_tree_if_exists(root / name, code_dir / name)

    snapshot_path = round_dir / "code_snapshot.tar.gz"
    with tarfile.open(snapshot_path, "w:gz") as tar:
        for name in ("FINAL_VERSION", "notes.md", "CMakeLists.txt", "README_AGENT.md", "kernels", "include", "src", "tools"):
            src = root / name
            if src.exists():
                tar.add(src, arcname=name)

    diff_path = round_dir / "diff_vs_previous.patch"
    if previous_round_dir and (previous_round_dir / "code").exists():
        proc = subprocess.run(
            [
                "diff",
                "-ruN",
                str(previous_round_dir / "code" / "kernels"),
                str(code_dir / "kernels"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        diff_path.write_text(proc.stdout or "", encoding="utf-8")
    else:
        diff_path.write_text("First continuation round; no previous code snapshot.\n", encoding="utf-8")


def kernel_manifest(root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted((root / "kernels").glob("gemm_v*.cu")):
        match = VERSION_RE.search(path.name)
        if not match:
            continue
        manifest.append(
            {
                "version": int(match.group(1)),
                "file": path.name,
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    return manifest


def write_all_versions_bundle(
    root: Path,
    rounds_dir: Path,
    round_dir: Path | None = None,
    round_idx: int | None = None,
) -> dict[str, Any]:
    versions = discover_versions(root)
    selected = selected_version(root, versions)
    all_versions_dir = rounds_dir / "all_versions"
    all_versions_dir.mkdir(parents=True, exist_ok=True)

    latest_archive = all_versions_dir / "all_versions_latest.tar.gz"
    with tarfile.open(latest_archive, "w:gz") as tar:
        for name in (
            "FINAL_VERSION",
            "notes.md",
            "CMakeLists.txt",
            "README_AGENT.md",
            "kernels",
            "include",
            "src",
            "tools",
        ):
            src = root / name
            if src.exists():
                tar.add(src, arcname=name)

    manifest = {
        "schema_version": 1,
        "task": "gemm-h100-kernel-lab",
        "created_at_utc": utc_now(),
        "round": round_idx,
        "root": str(root),
        "versions": versions,
        "version_count": len(versions),
        "selected_version": selected,
        "kernel_manifest": kernel_manifest(root),
        "archive": str(latest_archive),
    }
    write_json(all_versions_dir / "manifest_latest.json", manifest)
    (all_versions_dir / "latest_archive_path.txt").write_text(
        str(latest_archive) + "\n",
        encoding="utf-8",
    )

    if round_idx is not None:
        round_archive = all_versions_dir / f"all_versions_round_{round_idx:03d}.tar.gz"
        shutil.copy2(latest_archive, round_archive)
        round_manifest = dict(manifest)
        round_manifest["archive"] = str(round_archive)
        write_json(all_versions_dir / f"manifest_round_{round_idx:03d}.json", round_manifest)
        if round_dir is not None:
            copy_file_if_exists(round_archive, round_dir / "all_versions_snapshot.tar.gz")
            write_json(round_dir / "all_versions_manifest.json", round_manifest)

    return manifest


def compact_metrics(round_idx: int, tier: str, public_results: dict[str, Any] | None, root: Path) -> dict[str, Any]:
    versions = discover_versions(root)
    selected = selected_version(root, versions)
    if not public_results:
        return {
            "schema_version": 1,
            "round": round_idx,
            "tier": tier,
            "timestamp_utc": utc_now(),
            "candidate_version": selected,
            "versions": versions,
            "all_ok": False,
            "geomean_tflops": 0.0,
            "decision": "missing_public_results",
        }
    return {
        "schema_version": 1,
        "round": round_idx,
        "tier": tier,
        "timestamp_utc": public_results.get("timestamp_utc") or utc_now(),
        "eval_id": public_results.get("eval_id"),
        "candidate_version": public_results.get("candidate_version", selected),
        "versions": public_results.get("versions", versions),
        "candidate_families": public_results.get("candidate_families") or [],
        "completed_family_count": public_results.get("completed_family_count") or 0,
        "candidate_family_manifest": public_results.get("candidate_family_manifest") or [],
        "all_ok": bool(public_results.get("all_ok")),
        "geomean_tflops": fnum(public_results.get("geomean_tflops")),
        "history_best_version": public_results.get("history_best_version"),
        "history_best_tflops": fnum(public_results.get("history_best_tflops")),
        "public_rounds_since_history_best": public_results.get("public_rounds_since_history_best"),
        "public_feedback_rounds": public_results.get("public_feedback_rounds"),
        "decision": public_results.get("decision"),
        "shape_summaries": public_results.get("shape_summaries") or [],
    }


def metrics_rows(rounds_dir: Path) -> list[dict[str, Any]]:
    history_candidates = sorted(
        (path for path in rounds_dir.glob("round_*/public/public_history.jsonl") if path.is_file()),
        key=lambda path: (len(read_jsonl(path)), str(path)),
    )
    if history_candidates:
        state_rows = read_jsonl(rounds_dir / "continuation_state_by_round.jsonl")
        bounds: list[tuple[int, int]] = []
        for state in state_rows:
            native_round = iopt(state.get("round"))
            public_end = iopt(state.get("public_feedback_rounds"))
            if native_round is not None and public_end is not None:
                bounds.append((native_round, public_end))
        bounds.sort(key=lambda item: item[1])

        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(read_jsonl(history_candidates[-1]), start=1):
            public_round = iopt(row.get("public_feedback_rounds")) or iopt(row.get("eval_id")) or idx
            native_round = None
            for candidate_round, public_end in bounds:
                if public_round <= public_end:
                    native_round = candidate_round
                    break
            if native_round is None:
                native_round = bounds[-1][0] if bounds else public_round
            rows.append(
                {
                    "schema_version": 1,
                    "round": native_round,
                    "public_round": public_round,
                    "tier": row.get("mode") or "public",
                    "timestamp_utc": row.get("timestamp_utc"),
                    "eval_id": row.get("eval_id") or public_round,
                    "candidate_version": row.get("candidate_version"),
                    "versions": row.get("versions") or [],
                    "all_ok": bool(row.get("all_ok")),
                    "geomean_tflops": fnum(row.get("geomean_tflops")),
                    "history_best_version": row.get("history_best_version"),
                    "history_best_tflops": fnum(row.get("history_best_tflops")),
                    "public_rounds_since_history_best": row.get("public_rounds_since_history_best"),
                    "public_feedback_rounds": row.get("public_feedback_rounds") or public_round,
                    "decision": row.get("decision"),
                    "shape_summaries": row.get("shape_summaries") or [],
                }
            )
        return rows

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int | None, Any]] = set()
    for path in sorted((rounds_dir).glob("round_*/metrics.json")):
        data = read_json(path)
        if not data:
            continue
        key = (iopt(data.get("round")), data.get("eval_id"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(data)
    if not rows:
        rows = read_jsonl(rounds_dir / "training_metrics.jsonl")
    return sorted(rows, key=lambda row: (iopt(row.get("round")) or 0, iopt(row.get("eval_id")) or 0))


def write_analysis(rounds_dir: Path, benchmark_dir: Path) -> None:
    rows = metrics_rows(rounds_dir)
    analysis_dir = rounds_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    best = None
    valid = [row for row in rows if row.get("all_ok") and fnum(row.get("geomean_tflops")) > 0.0]
    if valid:
        best = max(valid, key=lambda row: fnum(row.get("geomean_tflops")))
    summary = {
        "schema_version": 1,
        "task": "gemm-h100-kernel-lab",
        "generated_at_utc": utc_now(),
        "round_count": len(rows),
        "best_round": best.get("round") if best else None,
        "best_version": best.get("candidate_version") if best else None,
        "best_public_tflops": fnum(best.get("geomean_tflops")) if best else 0.0,
        "latest_round": rows[-1] if rows else None,
    }
    write_json(analysis_dir / "summary.json", summary)

    csv_path = analysis_dir / "public_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "round",
                "public_round",
                "eval_id",
                "candidate_version",
                "geomean_tflops",
                "history_best_version",
                "history_best_tflops",
                "all_ok",
                "decision",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "round": row.get("round"),
                "public_round": row.get("public_round") or row.get("public_feedback_rounds") or row.get("eval_id"),
                "eval_id": row.get("eval_id"),
                "candidate_version": row.get("candidate_version"),
                "geomean_tflops": fnum(row.get("geomean_tflops")),
                "history_best_version": row.get("history_best_version"),
                "history_best_tflops": fnum(row.get("history_best_tflops")),
                "all_ok": row.get("all_ok"),
                "decision": row.get("decision") or "",
            })

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('round') or ''))}</td>"
            f"<td>{html.escape(str(row.get('public_round') or row.get('public_feedback_rounds') or row.get('eval_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('candidate_version') or ''))}</td>"
            f"<td>{fnum(row.get('geomean_tflops')):.6f}</td>"
            f"<td>{html.escape(str(row.get('history_best_version') or ''))}</td>"
            f"<td>{fnum(row.get('history_best_tflops')):.6f}</td>"
            f"<td>{html.escape(str(row.get('all_ok')))}</td>"
            f"<td>{html.escape(str(row.get('decision') or ''))}</td>"
            "</tr>"
        )
    html_path = analysis_dir / "public_curve.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>GEMM Public Continuation Curve</title>"
        "<style>body{font-family:system-ui, sans-serif;margin:24px;}"
        "table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;"
        "padding:6px 8px;text-align:left;}th{background:#f5f5f5;}"
        "code{background:#f3f3f3;padding:1px 4px;}</style></head><body>"
        "<h1>GEMM Public Continuation Curve</h1>"
        f"<p>Rounds: {len(rows)}. Best public TFLOPS: {summary['best_public_tflops']:.6f} "
        f"at v{html.escape(str(summary['best_version'] or ''))}.</p>"
        "<table><thead><tr><th>Native Round</th><th>Public Round</th><th>Version</th><th>Public TFLOPS</th>"
        "<th>History Best Version</th><th>History Best TFLOPS</th><th>OK</th>"
        "<th>Decision</th></tr></thead><tbody>"
        + ("\n".join(table_rows) or "<tr><td colspan='8'>No rounds yet.</td></tr>")
        + "</tbody></table></body></html>\n",
        encoding="utf-8",
    )

    latest_summary = [
        "# GEMM Continuation Summary",
        "",
        f"- Generated: {summary['generated_at_utc']}",
        f"- Rounds captured: {summary['round_count']}",
        f"- Best public version: {summary['best_version']}",
        f"- Best public TFLOPS: {summary['best_public_tflops']:.6f}",
        "",
        "## Latest Rounds",
        "",
        "| Native Round | Public Round | Version | Public TFLOPS | History Best | OK | Decision |",
        "| ---: | ---: | ---: | ---: | ---: | :---: | --- |",
    ]
    for row in rows[-8:]:
        latest_summary.append(
            f"| {row.get('round')} | "
            f"{row.get('public_round') or row.get('public_feedback_rounds') or row.get('eval_id')} | "
            f"{row.get('candidate_version')} | "
            f"{fnum(row.get('geomean_tflops')):.6f} | "
            f"{row.get('history_best_version')} / {fnum(row.get('history_best_tflops')):.6f} | "
            f"{'yes' if row.get('all_ok') else 'no'} | {row.get('decision') or ''} |"
        )
    latest_summary.extend([
        "",
        f"- Public curve CSV: `{csv_path}`",
        f"- Public curve HTML: `{html_path}`",
        "",
    ])
    latest_text = "\n".join(latest_summary)
    (rounds_dir / "latest_summary.md").write_text(latest_text, encoding="utf-8")
    training_dir = benchmark_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "latest_summary.md").write_text(latest_text, encoding="utf-8")
    copy_file_if_exists(csv_path, training_dir / "public_curve.csv")
    copy_file_if_exists(html_path, training_dir / "public_curve.html")
    copy_file_if_exists(analysis_dir / "summary.json", training_dir / "summary.json")
    copy_file_if_exists(rounds_dir / "training_metrics.jsonl", training_dir / "training_metrics.jsonl")
    copy_file_if_exists(rounds_dir / "public_results_by_round.jsonl", training_dir / "public_results_by_round.jsonl")
    copy_file_if_exists(rounds_dir / "continuation_state_by_round.jsonl", training_dir / "continuation_state_by_round.jsonl")
    copy_tree_if_exists(analysis_dir, training_dir / "analysis")
    copy_tree_if_exists(rounds_dir / "all_versions", training_dir / "all_versions")

    log_training_dir = rounds_dir.parent / "gemm_h100_kernel_lab_training"
    copy_tree_if_exists(training_dir, log_training_dir)


def previous_round_dir(rounds_dir: Path, round_idx: int) -> Path | None:
    candidates = []
    for path in rounds_dir.glob("round_*"):
        if not path.is_dir():
            continue
        match = re.search(r"round_([0-9]+)$", path.name)
        if not match:
            continue
        value = int(match.group(1))
        if value < round_idx:
            candidates.append((value, path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def command_eval(args: argparse.Namespace) -> int:
    app_dir = args.app_dir.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    rounds_dir = args.rounds_dir.resolve()
    root = gemm_root(app_dir)
    round_dir = rounds_dir / f"round_{args.native_session:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    public_stdout = ""
    public_exit = 0
    if args.reuse_latest:
        public_stdout = "Reused existing public_results.json; public feedback was not rerun.\n"
    else:
        try:
            public_exit, public_stdout = run_public_feedback(args)
        except subprocess.TimeoutExpired as exc:
            public_exit = 124
            timeout_stdout = exc.stdout or ""
            if isinstance(timeout_stdout, bytes):
                timeout_stdout = timeout_stdout.decode("utf-8", errors="replace")
            public_stdout = (
                timeout_stdout
                + f"\nPublic feedback timed out after {args.timeout}s.\n"
            )
    (round_dir / "public_stdout.txt").write_text(public_stdout, encoding="utf-8")
    (round_dir / "public_exit_code.txt").write_text(f"{public_exit}\n", encoding="utf-8")

    snapshot_public(benchmark_dir, round_dir)
    snapshot_code(root, round_dir, previous_round_dir(rounds_dir, args.native_session))
    all_versions_manifest = write_all_versions_bundle(
        root,
        rounds_dir,
        round_dir,
        args.native_session,
    )
    public_results = read_json(benchmark_dir / "public_results.json")
    metrics = compact_metrics(args.native_session, args.tier, public_results, root)
    metrics["public_exit_code"] = public_exit
    metrics["round_dir"] = str(round_dir)
    metrics["all_versions_archive"] = all_versions_manifest.get("archive")
    metrics["all_versions_count"] = all_versions_manifest.get("version_count")
    write_json(round_dir / "metrics.json", metrics)

    append_jsonl(rounds_dir / "training_metrics.jsonl", metrics)
    append_jsonl(rounds_dir / "public_results_by_round.jsonl", {
        "round": args.native_session,
        "round_dir": str(round_dir),
        "timestamp_utc": utc_now(),
        "public_exit_code": public_exit,
        "data": public_results or {},
    })
    write_analysis(rounds_dir, benchmark_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return public_exit


def command_status(args: argparse.Namespace) -> int:
    root = gemm_root(args.app_dir.resolve())
    versions = discover_versions(root)
    public_results = read_json(args.benchmark_dir / "public_results.json")
    rows = metrics_rows(args.rounds_dir)
    status = {
        "schema_version": 1,
        "task": "gemm-h100-kernel-lab",
        "timestamp_utc": utc_now(),
        "root": str(root),
        "versions": versions,
        "version_count": len(versions),
        "selected_version": selected_version(root, versions),
        "public_results": public_results,
        "captured_rounds": len(rows),
        "latest_round": rows[-1] if rows else None,
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def command_finalize_round(args: argparse.Namespace) -> int:
    app_dir = args.app_dir.resolve()
    root = gemm_root(app_dir)
    round_dir = args.rounds_dir / f"round_{args.native_session:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    if args.state_record_path and args.state_record_path.exists():
        copy_file_if_exists(args.state_record_path, round_dir / "continuation_state.json")
    snapshot_code(root, round_dir, previous_round_dir(args.rounds_dir, args.native_session))
    write_all_versions_bundle(root, args.rounds_dir, round_dir, args.native_session)
    write_analysis(args.rounds_dir, args.benchmark_dir)
    return 0


def command_finalize_session(args: argparse.Namespace) -> int:
    return command_finalize_round(args)


def command_rebuild_analysis(args: argparse.Namespace) -> int:
    write_analysis(args.rounds_dir, args.benchmark_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("/workspace/benchmark"))
    parser.add_argument("--rounds-dir", type=Path, default=Path("/logs/agent/continuation_rounds"))
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--native-session", type=int, required=True)
    eval_parser.add_argument("--tier", choices=["public", "smoke"], default="public")
    eval_parser.add_argument("--feedback-script", type=Path, default=Path("/workspace/benchmark/run_public_tests.sh"))
    eval_parser.add_argument("--timeout", type=int, default=int(os.environ.get("GEMM_PUBLIC_CONTROLLER_TIMEOUT", "1800")))
    eval_parser.add_argument("--reuse-latest", action="store_true")

    status_parser = sub.add_parser("status")
    status_parser.set_defaults(_status=True)

    finalize_parser = sub.add_parser("finalize-round")
    finalize_parser.add_argument("--native-session", type=int, required=True)
    finalize_parser.add_argument("--state-record-path", type=Path)

    finalize_session_parser = sub.add_parser("finalize-session")
    finalize_session_parser.add_argument("--native-session", type=int, required=True)
    finalize_session_parser.add_argument("--state-record-path", type=Path)

    sub.add_parser("rebuild-analysis")

    args = parser.parse_args()
    args.benchmark_dir = args.benchmark_dir.resolve()
    args.rounds_dir = args.rounds_dir.resolve()
    if args.command == "eval":
        return command_eval(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "finalize-round":
        return command_finalize_round(args)
    if args.command == "finalize-session":
        return command_finalize_session(args)
    if args.command == "rebuild-analysis":
        return command_rebuild_analysis(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
