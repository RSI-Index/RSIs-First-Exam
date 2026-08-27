from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from rsi_harness.cluster.bluevela.runtime import (
    ApptainerAgentRuntime,
    NativeJudgeEvaluator,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.models import AgentRunResult, ContainerRef, EvaluationRequest
from rsi_harness.runtime.artifacts import RunArtifactWriter
from tests.factories import make_run_plan


class _Runtime:
    def __init__(self, root: Path) -> None:
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "answer.py").write_text("answer = 42\n")
        self.rounds = root / "rounds"
        self.rounds.mkdir()
        self.payload = SimpleNamespace(run_id="native-run")
        self.events: list[str] = []

    def pause(self, _work: ContainerRef) -> None:
        self.events.append("pause")

    def unpause(self, _work: ContainerRef) -> None:
        self.events.append("unpause")

    def run_judge(self, workspace, request, environment):
        del workspace, environment
        (request.verifier_logs / "reward.txt").write_text("1\n")
        request.verifier_output.write_text("passed\n")
        return AgentRunResult(
            exit_code=0,
            output="passed\n",
            full_output_captured=True,
        )


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def resource_event(self, name: str, **values: object) -> None:
        self.events.append((name, values))


def test_workspace_snapshot_uses_path_authority_and_native_artifacts(
    tmp_path: Path,
) -> None:
    run_id = "native-run"
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id=run_id)
    writer.start()
    runtime = _Runtime(tmp_path)
    observer = _Observer()
    work = ContainerRef(container_id="work", role="work")
    request = EvaluationRequest(
        run_plan=plan,
        work_container=work,
        round_id="agent-1",
        verifier_logs=(
            writer.root / "verifier" / "agent-1"
        ),
        verifier_output=writer.feedback_root / "agent-1.log",
    )

    report = NativeJudgeEvaluator(runtime, writer).evaluate(request, observer)

    planned = dict(observer.events)["snapshot_planned"]
    assert "planned_snapshot_ref" not in planned
    assert report.score == 1.0
    assert runtime.events == ["pause", "unpause"]
    assert (writer.root / "submissions/agent-1/report.json").is_file()
    assert (writer.root / "feedback/agent-1.log").read_text() == "passed\n"


def test_judge_gets_fresh_writable_tmp_without_mutating_preserved_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )
    runtime.task_assets.mkdir(parents=True)
    (runtime.task_assets / "test_patch.diff").write_text("original\n")
    (runtime.task_assets / "setup_patch.diff").touch()
    workspace = runtime.rounds / "agent-1" / "workspace"
    workspace.mkdir(parents=True)
    request = EvaluationRequest(
        run_plan=plan,
        work_container=ContainerRef(container_id="work", role="work"),
        round_id="agent-1",
        verifier_logs=tmp_path / "verifier",
        verifier_output=tmp_path / "feedback.log",
    )

    def fake_run(command, **options):
        del command
        tmp_binds = [
            bind
            for bind in options["extra_binds"]
            if bind[1] == PurePosixPath("/tmp")
        ]
        assert len(tmp_binds) == 1
        judge_tmp, _target, read_only = tmp_binds[0]
        assert judge_tmp.parent == node_tmp / "judge"
        assert read_only is False
        assert (judge_tmp / "test_patch.diff").read_text() == "original\n"
        assert (judge_tmp / "setup_patch.diff").is_file()
        assert (judge_tmp / "setup_patch.diff").stat().st_size == 0
        (judge_tmp / "test_patch.diff").write_text("judge mutation\n")
        (judge_tmp / "f2p_output.txt").write_text("passed\n")
        return AgentRunResult(exit_code=0, output="passed\n")

    monkeypatch.setattr(runtime, "_run", fake_run)

    result = runtime.run_judge(workspace, request, {})

    assert result.exit_code == 0
    assert (runtime.task_assets / "test_patch.diff").read_text() == "original\n"
    assert not any((node_tmp / "judge").iterdir())


def test_apptainer_commands_use_run_plan_workdir(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )

    command = runtime._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        containall=True,
    )

    cwd_index = command.index("--cwd")
    assert command[cwd_index + 1] == "/workspace"
    assert (
        "--bind",
        f"{node_tmp / 'work'}:/tmp",
    ) in tuple(zip(command, command[1:], strict=False))
    environment = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for option, value in zip(command, command[1:], strict=False)
        if option == "--env"
    }
    assert environment["HF_HOME"] == str(runtime.profile.storage.hf_home)
    assert environment["HF_DATASETS_CACHE"] == str(
        runtime.profile.storage.hf_datasets_cache
    )
    assert environment["TMPDIR"] == "/tmp"
    assert environment["XDG_CACHE_HOME"] == "/tmp/.cache"
    assert environment["TRITON_CACHE_DIR"] == "/tmp/.cache/triton"
