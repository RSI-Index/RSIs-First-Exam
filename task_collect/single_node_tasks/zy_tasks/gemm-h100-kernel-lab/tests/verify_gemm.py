#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(
    r"GEMM_RESULT\b.*?\btflops=([0-9.eE+-]+).*?"
    r"\bmean_rel_error=([0-9.eE+-]+).*?"
    r"\bmax_rel_error=([0-9.eE+-]+).*?"
    r"\bbad_samples=([0-9]+).*?\bok=([01])"
)
VERSION_RE = re.compile(r"gemm_v([0-9]{3})(?:_|\.cu)")
LIB_PAT = re.compile(
    r"cutlass|cute/|#include\s*<cute|cublas|cublaslt|cudnn|triton|torch|rocblas|flashinfer|dlopen|dlsym",
    re.I,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reward(log_dir: Path, detail: dict[str, Any], reward: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    detail = dict(detail)
    detail["reward"] = float(reward)
    numeric: dict[str, float] = {"reward": float(reward)}
    for key, value in detail.items():
        if isinstance(value, bool):
            numeric[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                numeric[key] = float(value)
    (log_dir / "gemm_results.json").write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (log_dir / "reward.json").write_text(json.dumps(numeric, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (log_dir / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    print(json.dumps(detail, indent=2, sort_keys=True))


def fail(log_dir: Path, message: str, extra: dict[str, Any] | None = None) -> int:
    detail: dict[str, Any] = {
        "correctness": 0.0,
        "protected_ok": 0.0,
        "handwritten_ok": 0.0,
        "geomean_tflops": 0.0,
        "error": message,
    }
    if extra:
        detail.update(extra)
    write_reward(log_dir, detail, 0.0)
    return 1


def parse_shapes(text: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for chunk in text.split(","):
        parts = chunk.lower().strip().split("x")
        if len(parts) != 3:
            continue
        shapes.append(tuple(int(part) for part in parts))
    return shapes


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


def resolve_protected_path(app_dir: Path, raw_path: str) -> Path:
    if raw_path.startswith("/"):
        path = Path(raw_path)
        if path.exists():
            return path
        workspace_prefix = "/workspace/benchmark/"
        if raw_path.startswith(workspace_prefix):
            suffix = raw_path[len(workspace_prefix):]
            for candidate in (
                app_dir / "benchmark" / suffix,
                app_dir / "environment" / "benchmark" / suffix,
            ):
                if candidate.exists():
                    return candidate
            return app_dir / "environment" / "benchmark" / suffix
        return path
    direct = app_dir / raw_path
    if direct.exists():
        return direct
    environment_path = app_dir / "environment" / raw_path
    if environment_path.exists():
        return environment_path
    return direct


def verify_protected_hashes(app_dir: Path, tests_dir: Path) -> list[str]:
    manifest = tests_dir / "protected_hashes.txt"
    if not manifest.exists():
        return [f"missing protected hash manifest: {manifest}"]
    problems: list[str] = []
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, rel = line.split(None, 1)
        path = resolve_protected_path(app_dir, rel)
        if not path.exists():
            problems.append(f"missing protected file: {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            problems.append(f"protected file changed: {rel}")
    return problems


def strip_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def source_scan(root: Path) -> list[str]:
    problems: list[str] = []
    for path in (root / "kernels").rglob("*"):
        if not path.is_file() or path.suffix not in {".cu", ".cuh", ".h", ".hpp", ".cpp"}:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if LIB_PAT.search(strip_line_comment(line)):
                problems.append(f"{path.relative_to(root)}:{lineno}: prohibited library/framework reference")
    return problems


def run_cmd(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda")},
    )
    return proc.returncode, proc.stdout


def build_candidate(root: Path, version: int, args: argparse.Namespace) -> tuple[int, Path | None, str]:
    rc, out = run_cmd(
        ["./tools/build.sh", "--version", str(version)],
        root,
        args.build_timeout,
    )
    if rc != 0:
        return rc, None, out
    binary_line = ""
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line:
            binary_line = line
            break
    if not binary_line:
        return 1, None, out + "\nmissing build output binary path\n"
    binary = Path(binary_line)
    if not binary.is_absolute():
        binary = root / binary
    return rc, binary, out


def build_failure_row(
    version: int,
    shape: tuple[int, int, int],
    repeat: int,
    output: str,
) -> dict[str, Any]:
    m, n, k = shape
    return {
        "version": version,
        "shape": [m, n, k],
        "repeat": repeat,
        "rc": 1,
        "elapsed_sec": 0.0,
        "ok": False,
        "error": "build failed",
        "output_tail": output[-6000:],
    }


def run_one(
    root: Path,
    binary: Path,
    version: int,
    shape: tuple[int, int, int],
    repeat: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    m, n, k = shape
    seed = args.seed + repeat * 65537 + m * 13 + n * 17 + k * 19
    cmd = [
        str(binary),
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--samples",
        str(args.samples),
        "--seed",
        str(seed),
    ]
    started = time.time()
    rc, out = run_cmd(cmd, root, args.run_timeout)
    row: dict[str, Any] = {
        "version": version,
        "shape": [m, n, k],
        "repeat": repeat,
        "rc": rc,
        "elapsed_sec": time.time() - started,
        "ok": False,
    }
    match = RESULT_RE.search(out)
    if rc != 0 or not match:
        row["error"] = "run failed or result line missing"
        row["output_tail"] = out[-6000:]
        return row
    row.update(
        {
            "tflops": float(match.group(1)),
            "mean_rel_error": float(match.group(2)),
            "max_rel_error": float(match.group(3)),
            "bad_samples": int(match.group(4)),
            "ok": match.group(5) == "1",
        }
    )
    return row


def geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    product = 1.0
    for value in values:
        product *= max(value, 1e-12)
    return product ** (1.0 / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument("--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier")))
    parser.add_argument(
        "--shapes",
        default=os.environ.get(
            "GEMM_VERIFY_SHAPES",
            "4096x4096x4096",
        ),
    )
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("GEMM_VERIFY_WARMUP", "20")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("GEMM_VERIFY_ITERS", "200")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("GEMM_VERIFY_REPEATS", "5")))
    parser.add_argument("--samples", type=int, default=int(os.environ.get("GEMM_VERIFY_SAMPLES", "2048")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("GEMM_VERIFY_SEED", "123456789")))
    parser.add_argument("--run-timeout", type=int, default=int(os.environ.get("GEMM_VERIFY_RUN_TIMEOUT", "2400")))
    parser.add_argument("--build-timeout", type=int, default=int(os.environ.get("GEMM_VERIFY_BUILD_TIMEOUT", "900")))
    args = parser.parse_args()

    app_dir = args.app_dir.resolve()
    tests_dir = Path(__file__).resolve().parent
    try:
        root = gemm_root(app_dir)
    except Exception as exc:
        return fail(args.log_dir, str(exc))

    protected = verify_protected_hashes(app_dir, tests_dir)
    if protected:
        return fail(args.log_dir, "protected files changed", {"protected_failures": protected})

    handwritten = source_scan(root)
    if handwritten:
        return fail(args.log_dir, "candidate uses prohibited library/framework references", {"handwritten_problems": handwritten, "protected_ok": 1.0})

    versions = discover_versions(root)
    version = selected_version(root, versions)
    if version is None:
        return fail(args.log_dir, "no candidate version found", {"versions": versions, "protected_ok": 1.0, "handwritten_ok": 1.0})
    if version not in versions:
        return fail(args.log_dir, "FINAL_VERSION does not match a candidate file", {"final_version": version, "versions": versions, "protected_ok": 1.0, "handwritten_ok": 1.0})

    shapes = parse_shapes(args.shapes)
    if not shapes:
        return fail(args.log_dir, "no verifier shapes configured", {"protected_ok": 1.0, "handwritten_ok": 1.0})

    build_rc, binary, build_out = build_candidate(root, version, args)
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        for repeat in range(1, args.repeats + 1):
            if build_rc != 0 or binary is None:
                rows.append(build_failure_row(version, shape, repeat, build_out))
            else:
                rows.append(run_one(root, binary, version, shape, repeat, args))

    failed = [row for row in rows if not row.get("ok")]
    shape_summaries: list[dict[str, Any]] = []
    medians: list[float] = []
    for shape in shapes:
        shape_rows = [row for row in rows if row["shape"] == list(shape)]
        valid = [float(row["tflops"]) for row in shape_rows if row.get("ok") and "tflops" in row]
        median = statistics.median(valid) if valid else 0.0
        mean = statistics.mean(valid) if valid else 0.0
        if median > 0.0:
            medians.append(median)
        shape_summaries.append(
            {
                "shape": list(shape),
                "run_count": len(shape_rows),
                "ok_runs": len(valid),
                "median_tflops": median,
                "mean_tflops": mean,
            }
        )

    if failed:
        return fail(
            args.log_dir,
            "candidate failed build, run, or correctness on at least one verifier run",
            {
                "protected_ok": 1.0,
                "handwritten_ok": 1.0,
                "final_version": version,
                "versions": versions,
                "benchmark_runs": rows,
                "failed_runs": failed,
            },
        )

    score = statistics.mean(medians) if medians else 0.0
    geometric_mean_tflops = geomean(medians)
    detail: dict[str, Any] = {
        "schema_version": 1,
        "protected_ok": 1.0,
        "handwritten_ok": 1.0,
        "correctness": 1.0,
        "final_version": float(version),
        "num_candidate_versions": float(len(versions)),
        "geomean_tflops": score,
        "average_tflops": score,
        "geometric_mean_tflops": geometric_mean_tflops,
        "score_tflops": score,
        "score_method": "median TFLOPS on hidden-seed 4096x4096x4096",
        "reward_definition": "reward = median TFLOPS on hidden-seed 4096x4096x4096 after protected-file, source-scan, and correctness gates",
        "shapes": [list(shape) for shape in shapes],
        "warmup": float(args.warmup),
        "iters": float(args.iters),
        "samples": float(args.samples),
        "repeats": float(args.repeats),
        "shape_summaries": shape_summaries,
        "benchmark_runs": rows,
    }
    write_reward(args.log_dir, detail, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
