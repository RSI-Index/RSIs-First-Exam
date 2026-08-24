from __future__ import annotations

import hashlib
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.config import ClusterProfile, load_cluster_profile
from rsi_harness.cluster.schedulers.lsf import (
    LSFJobResult,
    LSFJobSpec,
    LSFScheduler,
)
from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    FrozenRewardMap,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter

TARGET = (
    "linkedin__liger-kernel.c856fbab."
    "test_fused_neighborhood_attention.78217be4.lv2"
)


def _profile(tmp_path: Path) -> ClusterProfile:
    base = load_cluster_profile("bluevela", {"USER": "alice"})
    root = (tmp_path / "cluster").resolve()
    apptainer = tmp_path / "apptainer"
    apptainer.write_text("#!/bin/sh\nexit 0\n")
    apptainer.chmod(0o700)
    return base.model_copy(
        update={
            "storage": base.storage.model_copy(
                update={
                    "run_root": root / "runs",
                    "image_cache": root / "images",
                    "logs_root": root / "logs",
                    "hf_home": root / "hf",
                    "hf_datasets_cache": root / "datasets",
                }
            ),
            "builder": base.builder.model_copy(update={"temp_root": tmp_path}),
            "apptainer": base.apptainer.model_copy(
                update={"binary": apptainer}
            ),
        }
    )


def _request(tmp_path: Path, *, dry_run: bool) -> ClusterRunRequest:
    root = Path(__file__).resolve().parents[3]
    return ClusterRunRequest(
        task_dir=(root / "sample_tasks" / TARGET).resolve(),
        agent_name="codex",
        options=CompileOptions(agent_name="codex"),
        logs_root=(tmp_path / "ignored-local-logs").resolve(),
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        agent_auth=AgentAuthSource.LOCAL,
        dry_run=dry_run,
    )


class RecordingScheduler:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.specs: list[LSFJobSpec] = []
        self.renderer = LSFScheduler()

    def render_submit(self, spec: LSFJobSpec) -> tuple[str, ...]:
        return self.renderer.render_submit(spec)

    def require_name_available(self, name: str) -> None:
        stage = "build" if "-build-" in name else "run"
        self.events.append(f"check-{stage}")

    def submit(self, spec: LSFJobSpec) -> str:
        self.specs.append(spec)
        stage = "build" if spec.gpu_count == 0 else "run"
        self.events.append(f"submit-{stage}")
        return str(100 + len(self.specs))

    def wait(self, job_id: str, **_kwargs) -> LSFJobResult:
        spec = self.specs[int(job_id) - 101]
        stage = "build" if spec.gpu_count == 0 else "run"
        self.events.append(f"wait-{stage}")
        if stage == "build":
            assignments = {
                line.split("=", 1)[0]: shlex.split(line.split("=", 1)[1])[0]
                for line in spec.script_path.read_text().splitlines()
                if line.startswith(("final_sif=", "final_sha="))
            }
            sif = Path(assignments["final_sif"])
            sha = Path(assignments["final_sha"])
            sif.parent.mkdir(parents=True, exist_ok=True)
            sif.write_bytes(b"test-sif")
            digest = hashlib.sha256(b"test-sif").hexdigest()
            sha.write_text(f"{digest}  {sif.name}\n")
        else:
            payload = load_engine_payload(spec.script_path.with_suffix(".json"))
            writer = RunArtifactWriter(payload.run_plan, run_id=payload.run_id)
            writer.start()
            (writer.root / "agent_prompt.md").write_text(
                payload.run_plan.task.instruction
            )
            (writer.root / "agent_output.txt").write_text("agent output\n")
            (writer.root / "run_agent.log").write_text("trial log\n")
            writer.record_submission(
                SubmissionReport(
                    round_id="agent-1",
                    status=SubmissionStatus.COMPLETED,
                    rewards=FrozenRewardMap({"reward": 1.0}),
                    score=1.0,
                    output="passed\n",
                )
            )
            writer.finalize()
        return LSFJobResult(job_id=job_id, state="DONE", exit_code=0)


def test_dry_run_resolves_four_gpus_without_mutation(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    events: list[tuple[str, object]] = []
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        clock=lambda: datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.149.0",
    )

    result = adapter.run(_request(tmp_path, dry_run=True))

    assert result.job_ids == ()
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()
    dry_run = dict(events)["dry_run"]
    assert dry_run["resources"] == {
        "work_gpus": 2,
        "verifier_gpus": 2,
        "total_gpus": 4,
        "cpu_slots": 8,
        "memory_mb": 65536,
        "local_tmp_mb": 15360,
        "build_walltime": "02:00",
        "run_walltime": "02:15",
    }
    assert "num=4:mode=exclusive_process" in dry_run["run_argv"]
    assert "select[tmp>=15360] span[hosts=1]" in dry_run["run_argv"]
    assert dry_run["agent_version"] == "0.149.0"


def test_cache_miss_waits_for_build_then_run_and_records_manifest(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        clock=lambda: datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        agent_version_resolver=lambda _name: "0.149.0",
    )

    result = adapter.run(_request(tmp_path, dry_run=False))

    assert scheduler.events == [
        "check-build",
        "submit-build",
        "wait-build",
        "check-run",
        "submit-run",
        "wait-run",
    ]
    assert result.job_ids == ("101", "102")
    assert scheduler.specs[0].gpu_count == 0
    assert scheduler.specs[0].local_tmp_mb == profile.builder.min_tmp_mb
    assert scheduler.specs[1].gpu_count == 4
    assert scheduler.specs[1].local_tmp_mb == 15360
    payload = load_engine_payload(scheduler.specs[1].script_path.with_suffix(".json"))
    assert payload.source_root.parent.name == "control"
    assert payload.source_root.name == "engine-source"
    assert (payload.source_root / "src/rsi_harness/__init__.py").is_file()
    assert result.log_dir.is_dir()
    manifest = json.loads(
        (profile.storage.run_root / result.run_id / "RUN_INFO.json").read_text()
    )
    assert manifest["build_job_id"] == "101"
    assert manifest["run_job_id"] == "102"
    assert manifest["resources"]["total_gpus"] == 4
    assert manifest["state"] == "completed"
    assert "auth.json" not in json.dumps(manifest)
