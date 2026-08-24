"""Native RSI-Harness Engine execution inside one Blue Vela allocation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from pydantic import field_validator

from rsi_harness.cluster.bluevela.adapter import ClusterResources
from rsi_harness.cluster.config import ClusterProfile
from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    GPUAllocation,
    GPUDevice,
    PersistedModel,
    RunGPUPlan,
    RunPlan,
)


class EnginePayload(PersistedModel):
    """Frozen inputs consumed by the compute-node RSI-Harness Engine."""

    run_id: str
    run_plan: RunPlan
    sif_path: Path
    sif_sha256_path: Path
    source_root: Path
    resources: ClusterResources
    profile: ClusterProfile
    options: CompileOptions
    agent_auth: AgentAuthSource | None = None
    agent_version: str | None = None
    agent_binary: Path | None = None
    agent_companions: tuple[Path, ...] = ()

    @field_validator(
        "sif_path", "sif_sha256_path", "source_root", "agent_binary"
    )
    @classmethod
    def _absolute(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if not value.is_absolute():
            raise ValueError("cluster Engine payload paths must be absolute")
        return value

    @field_validator("agent_companions")
    @classmethod
    def _absolute_companions(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if any(not path.is_absolute() for path in value):
            raise ValueError("cluster Agent companion paths must be absolute")
        return value


def load_engine_payload(path: Path) -> EnginePayload:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SetupError(f"invalid cluster Engine payload {path}: {error}") from error
    return EnginePayload.model_validate(value)


def bind_lsf_devices(
    plan: RunPlan,
    allocated_devices: tuple[str, ...],
) -> RunPlan:
    """Replace planning placeholders with the exact ordered LSF allocation."""
    expected = len(plan.gpu_plan.authorized_pool.devices)
    if (
        len(allocated_devices) != expected
        or any(not device for device in allocated_devices)
        or len(set(allocated_devices)) != len(allocated_devices)
    ):
        raise SetupError(
            "LSF CUDA_VISIBLE_DEVICES does not match the Engine plan: "
            f"expected {expected}, got {allocated_devices!r}"
        )
    devices = tuple(
        GPUDevice(index=index, uuid=value, name="LSF allocated GPU")
        for index, value in enumerate(allocated_devices)
    )
    work_count = len(plan.gpu_plan.work.devices)
    judge_count = len(plan.gpu_plan.judge.devices)
    if work_count + judge_count != expected:
        raise SetupError("Engine Work/Judge GPU slices do not cover the allocation")
    gpu_plan = RunGPUPlan(
        authorized_pool=GPUAllocation(devices=devices),
        work=GPUAllocation(devices=devices[:work_count]),
        judge=GPUAllocation(devices=devices[work_count:]),
        judge_mode=plan.gpu_plan.judge_mode,
    )
    return plan.model_copy(update={"gpu_plan": gpu_plan})


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def render_engine_driver(
    payload: EnginePayload,
    profile: ClusterProfile,
    output_path: Path,
) -> Path:
    """Render a GPU driver that starts the native Engine, never Harbor CLI."""
    output = Path(output_path).resolve()
    payload_path = output.with_suffix(".json")
    payload_path.write_text(payload.model_dump_json(indent=2))
    leaf = (
        payload.run_plan.paths.logs
        / "runs"
        / payload.run_id
        / payload.run_plan.task.task_id
    )
    binds = [
        item
        for path in profile.apptainer.extra_binds
        for item in ("--bind", f"{path}:{path}")
    ]
    bind_args = " ".join(_quote(item) for item in binds)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077

test -n "${{CUDA_VISIBLE_DEVICES:-}}"
export APPTAINER_BIND={_quote(profile.apptainer.dns_bind)}
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export HF_HOME={_quote(profile.storage.hf_home)}
export HF_DATASETS_CACHE={_quote(profile.storage.hf_datasets_cache)}
export PYTHONPATH={_quote(payload.source_root / 'src')}
runtime_tmp_root={_quote(profile.apptainer.temp_root)}
available_tmp_kb="$(df -Pk -- "$runtime_tmp_root" | awk 'NR == 2 {{print $4}}')"
required_tmp_kb=$(({payload.resources.local_tmp_mb} * 1024))
if [[ ! "$available_tmp_kb" =~ ^[0-9]+$ ]] || \
   (( available_tmp_kb < required_tmp_kb )); then
  echo "insufficient node-local runtime scratch:" \
    "need {payload.resources.local_tmp_mb} MiB under $runtime_tmp_root," \
    "have ${{available_tmp_kb:-unknown}} KiB" >&2
  exit 1
fi
export RSI_HARNESS_NODE_TMP
RSI_HARNESS_NODE_TMP="$(mktemp -d "$runtime_tmp_root/rsi-harness-engine.XXXXXX")"
cleanup() {{
  rm -rf -- "$RSI_HARNESS_NODE_TMP"
}}
trap cleanup EXIT

(cd {_quote(payload.sif_path.parent)} && \
  sha256sum -c {_quote(payload.sif_sha256_path.name)})
{_quote(profile.apptainer.binary)} exec --nv --containall --writable-tmpfs \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" {bind_args} \
  {_quote(payload.sif_path)} nvidia-smi

{_quote(sys.executable)} -m rsi_harness.cluster.bluevela.engine \
  --payload {_quote(payload_path)}
test -s {_quote(leaf / 'final_result.json')}
"""
    output.write_text(script)
    output.chmod(0o700)
    return payload_path


def run_engine_payload(payload: EnginePayload) -> None:
    """Run the native RSI-Harness Engine inside the current LSF allocation."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    plan = bind_lsf_devices(payload.run_plan, devices)
    from rsi_harness.cluster.bluevela.runtime import run_native_engine

    run_native_engine(payload, plan)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    run_engine_payload(load_engine_payload(args.payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
