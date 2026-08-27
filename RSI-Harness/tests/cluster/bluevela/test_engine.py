from __future__ import annotations

from pathlib import Path

from rsi_harness.cluster.bluevela.adapter import ClusterResources
from rsi_harness.cluster.bluevela.engine import (
    EnginePayload,
    bind_lsf_devices,
    render_engine_driver,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.models import AgentAuthSource, CompileOptions
from tests.factories import make_run_plan


def _payload(tmp_path: Path) -> EnginePayload:
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    plan = make_run_plan(tmp_path)
    devices = plan.gpu_plan.authorized_pool.devices
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "authorized_pool": plan.gpu_plan.authorized_pool.model_copy(
                        update={"devices": devices * 2}
                    ),
                    "work": plan.gpu_plan.work.model_copy(
                        update={"devices": devices}
                    ),
                    "judge": plan.gpu_plan.judge.model_copy(
                        update={"devices": devices[1:] or devices}
                    ),
                }
            )
        }
    )
    # Use four distinct placeholder devices, matching a 2+2 task allocation.
    from rsi_harness.models import GPUAllocation, GPUDevice, RunGPUPlan

    all_devices = tuple(
        GPUDevice(index=index, uuid=f"LSF-{index}", name="allocated")
        for index in range(4)
    )
    plan = plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=all_devices),
                work=GPUAllocation(devices=all_devices[:2]),
                judge=GPUAllocation(devices=all_devices[2:]),
                judge_mode="disjoint",
            )
        }
    )
    return EnginePayload(
        run_id="native-engine-run",
        run_plan=plan,
        sif_path=(tmp_path / "task.sif").resolve(),
        sif_sha256_path=(tmp_path / "task.sif.sha256").resolve(),
        source_root=Path(__file__).resolve().parents[3],
        resources=ClusterResources(
            work_gpus=2,
            verifier_gpus=2,
            total_gpus=4,
            cpu_slots=8,
            memory_mb=65536,
            local_tmp_mb=15360,
            build_walltime="02:00",
            run_walltime="02:15",
        ),
        profile=profile,
        options=CompileOptions(max_submissions=4),
        agent_auth=AgentAuthSource.LOCAL,
        agent_version="0.149.0",
        agent_binary=Path("/bin/true"),
        agent_companions=(Path("/bin/false"),),
    )


def test_lsf_devices_replace_placeholders_and_remain_disjoint(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    plan = bind_lsf_devices(payload.run_plan, ("0", "2", "5", "7"))

    assert plan.gpu_plan.authorized_pool.uuids == ("0", "2", "5", "7")
    assert plan.gpu_plan.work.uuids == ("0", "2")
    assert plan.gpu_plan.judge.uuids == ("5", "7")


def test_gpu_driver_invokes_native_engine_without_harbor_run(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    output = (tmp_path / "run.sh").resolve()

    payload_path = render_engine_driver(payload, profile, output)

    script = output.read_text()
    assert "rsi_harness.cluster.bluevela.engine" in script
    assert "harbor run" not in script
    assert "harbor-jobs" not in script
    assert "RSI_HARNESS_NODE_TMP" in script
    assert "available_tmp_kb" in script
    assert "required_tmp_kb=$((15360 * 1024))" in script
    assert payload_path == output.with_suffix(".json")


def test_engine_payload_keeps_native_submission_controls(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    assert payload.options.max_submissions == 4
    assert payload.run_plan.paths.logs.is_absolute()
    assert payload.run_plan.task.verifier.command == ("/bin/bash", "/tests/test.sh")
