"""Blue Vela resource planning and run orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import Field

from rsi_harness.cluster.base import (
    ClusterAdapter,
    ClusterRunRequest,
    ClusterRunResult,
)
from rsi_harness.cluster.bluevela.image import ImagePlan as SIFImagePlan
from rsi_harness.cluster.bluevela.image import (
    plan_image,
    render_build_driver,
    validate_cached_image,
)
from rsi_harness.cluster.config import (
    ClusterProfile,
    load_cluster_profile,
)
from rsi_harness.cluster.schedulers.lsf import (
    LSFJobResult,
    LSFJobSpec,
    LSFScheduler,
)
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    GPUAllocation,
    GPUDevice,
    ImagePlan,
    JudgeGPUMode,
    PersistedModel,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
    RunPlan,
    RunStatus,
    TaskDefinition,
)
from rsi_harness.runtime.redaction import redact_text
from rsi_harness.task.compiler import HarborTaskCompiler


class ClusterResources(PersistedModel):
    work_gpus: int = Field(ge=0)
    verifier_gpus: int = Field(ge=0)
    total_gpus: int = Field(ge=0)
    cpu_slots: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    local_tmp_mb: int = Field(ge=0)
    build_walltime: str
    run_walltime: str


def _lsf_walltime(seconds: float) -> str:
    minutes = max(1, math.ceil(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def _lsf_walltime_seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+):([0-5]\d)", value)
    if match is None:
        raise SetupError(
            f"cluster builder walltime must use HH:MM format, got {value!r}"
        )
    hours, minutes = (int(part) for part in match.groups())
    total = (hours * 60 + minutes) * 60
    if total == 0:
        raise SetupError("cluster builder walltime must be positive")
    return total


def derive_resources(
    definition: TaskDefinition,
    profile: ClusterProfile,
) -> ClusterResources:
    """Derive one-node allocation requirements from a compiled Harbor task."""
    declared_work = definition.gpu_requirement.count
    if declared_work == "all":
        override = profile.resources.all_gpus_override
        if override is None:
            raise SetupError(
                "task declares gpus='all'; cluster profile needs a numeric override"
            )
        work_gpus = override
    else:
        work_gpus = declared_work

    verifier_gpus = definition.verifier.gpu_count
    total_gpus = work_gpus + verifier_gpus
    if total_gpus > profile.resources.gpus_per_node:
        raise SetupError(
            "task GPU request exceeds cluster single-node capacity: "
            f"{work_gpus}+{verifier_gpus}={total_gpus} > "
            f"{profile.resources.gpus_per_node}"
        )

    run_seconds = (
        definition.agent.timeout_seconds
        + definition.verifier.timeout_seconds
        + profile.resources.walltime_margin_seconds
    )
    build_seconds = max(
        _lsf_walltime_seconds(profile.builder.walltime),
        definition.service.build_timeout_seconds
        + profile.resources.walltime_margin_seconds,
    )
    return ClusterResources(
        work_gpus=work_gpus,
        verifier_gpus=verifier_gpus,
        total_gpus=total_gpus,
        cpu_slots=max(profile.resources.min_cpu_slots, definition.service.cpus or 1),
        memory_mb=max(
            profile.resources.min_memory_mb,
            definition.service.memory_mb or 1,
        ),
        local_tmp_mb=definition.service.storage_mb or 0,
        build_walltime=_lsf_walltime(build_seconds),
        run_walltime=_lsf_walltime(run_seconds),
    )


class SchedulerPort(Protocol):
    def render_submit(self, spec: LSFJobSpec) -> tuple[str, ...]: ...

    def require_name_available(self, name: str) -> None: ...

    def submit(self, spec: LSFJobSpec) -> str: ...

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float,
        on_state: Callable[[str], None] | None = None,
    ) -> LSFJobResult: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_component(value: str, *, limit: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or "task")[:limit].rstrip("-")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resolve_agent_version(agent_name: str) -> str | None:
    if agent_name != "codex":
        return None
    executable = shutil.which(agent_name)
    if executable is None:
        raise SetupError(f"cluster agent executable is unavailable: {agent_name}")
    completed = subprocess.run(
        (executable, "--version"),
        check=False,
        capture_output=True,
        text=True,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r"\b(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b", output)
    if completed.returncode != 0 or match is None:
        raise SetupError(f"cannot resolve {agent_name} version from {output.strip()!r}")
    return match.group(1)


class BlueVelaClusterAdapter(ClusterAdapter):
    """Synchronous LSF/Apptainer transport for the native Harness Engine."""

    def __init__(
        self,
        profile: ClusterProfile,
        *,
        scheduler: SchedulerPort | None = None,
        compiler: HarborTaskCompiler | None = None,
        clock: Callable[[], datetime] = _utc_now,
        event_callback: Callable[[str, object], None] | None = None,
        source_root: Path | None = None,
        agent_version_resolver: Callable[[str], str | None] = _resolve_agent_version,
    ) -> None:
        self.profile = profile
        self.scheduler = scheduler or LSFScheduler(
            submit_binary=profile.scheduler.submit_binary,
            status_binary=profile.scheduler.status_binary,
            cancel_binary=profile.scheduler.cancel_binary,
        )
        self.compiler = compiler or HarborTaskCompiler()
        self.clock = clock
        self.event_callback = event_callback or (lambda _name, _value: None)
        self.source_root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[4]
        )
        self.agent_version_resolver = agent_version_resolver

    def run(self, request: ClusterRunRequest) -> ClusterRunResult:
        definition = self._compile(request)
        resources = derive_resources(definition, self.profile)
        build_context = definition.service.build_context
        if build_context is None:
            raise SetupError("Blue Vela cluster runs require a Docker build context")
        self._validate_runtime_inputs(request)
        agent_version = self.agent_version_resolver(definition.agent.name)
        raw_agent_binary = shutil.which(definition.agent.name)
        agent_binary = (
            None if raw_agent_binary is None else Path(raw_agent_binary).resolve()
        )
        agent_companions = ()
        if agent_binary is not None:
            code_mode_host = agent_binary.parent / "codex-code-mode-host"
            if code_mode_host.is_file():
                agent_companions = (code_mode_host.resolve(),)

        run_id = self._run_id(definition.task_id)
        run_dir = self.profile.storage.run_root / run_id
        names = {
            stage: self._job_name(definition.task_id, stage)
            for stage in ("build", "run")
        }
        planned_image = plan_image(build_context, self.profile.storage.image_cache)
        build_spec = self._build_spec(
            run_dir, names["build"], resources, run_dir / "control" / "build.sh"
        )
        run_spec = self._run_spec(
            run_dir, names["run"], resources, run_dir / "control" / "run.sh"
        )
        if request.dry_run:
            self.event_callback(
                "dry_run",
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "image": str(planned_image.sif_path),
                    "cache_hit": planned_image.cache_hit,
                    "resources": resources.model_dump(mode="json"),
                    "build_argv": self.scheduler.render_submit(build_spec),
                    "run_argv": self.scheduler.render_submit(run_spec),
                    "binds": tuple(
                        str(path) for path in self.profile.apptainer.extra_binds
                    ),
                    "agent_version": agent_version,
                },
            )
            return ClusterRunResult(
                run_id=run_id,
                status=RunStatus.PREPARING,
                log_dir=self.profile.storage.logs_root / "runs" / run_id,
            )

        run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        (run_dir / "build").mkdir()
        (run_dir / "run").mkdir()
        control = run_dir / "control"
        control.mkdir()
        frozen_task, frozen_source = self._freeze_control(request.task_dir, control)
        definition = self._relocate_definition(definition, frozen_task)
        image = plan_image(
            definition.service.build_context,
            self.profile.storage.image_cache,
        )
        render_build_driver(
            image,
            self.profile,
            definition.service.build_context,
            build_spec.script_path,
            build_spec.local_tmp_mb,
        )
        manifest = self._manifest(
            run_id=run_id,
            run_dir=run_dir,
            definition=definition,
            resources=resources,
            image=image,
            agent_version=agent_version,
        )
        manifest_path = run_dir / "RUN_INFO.json"
        _atomic_json(manifest_path, manifest)
        job_ids: list[str] = []
        try:
            if not image.cache_hit:
                self.scheduler.require_name_available(build_spec.name)
                build_job_id = self.scheduler.submit(build_spec)
                job_ids.append(build_job_id)
                manifest["build_job_id"] = build_job_id
                manifest["state"] = "building"
                _atomic_json(manifest_path, manifest)
                self.event_callback(
                    "job_submitted", {"stage": "build", "job_id": build_job_id}
                )
                build_result = self.scheduler.wait(
                    build_job_id,
                    poll_seconds=self.profile.scheduler.poll_seconds,
                    on_state=lambda state: self.event_callback(
                        "job_state", {"stage": "build", "state": state}
                    ),
                )
                self._require_success("SIF build", build_result)
            if not validate_cached_image(image):
                raise InfrastructureError(
                    "SIF cache is missing or failed SHA256 validation after build"
                )

            run_plan = self._run_plan(definition, image, resources, run_dir)
            from rsi_harness.cluster.bluevela.engine import (
                EnginePayload,
                render_engine_driver,
            )
            from rsi_harness.cluster.bluevela.runtime import validate_native_artifacts

            payload = EnginePayload(
                run_id=run_id,
                run_plan=run_plan,
                sif_path=image.sif_path,
                sif_sha256_path=image.sha256_path,
                source_root=frozen_source,
                resources=resources,
                profile=self.profile,
                options=request.options,
                agent_auth=request.agent_auth,
                agent_version=agent_version,
                agent_binary=agent_binary,
                agent_companions=agent_companions,
            )
            render_engine_driver(payload, self.profile, run_spec.script_path)
            manifest["sif_sha256"] = image.sha256_path.read_text().split()[0]
            manifest["control_sha256"] = self._control_hashes(control)
            manifest["state"] = "ready"
            _atomic_json(manifest_path, manifest)

            self.scheduler.require_name_available(run_spec.name)
            run_job_id = self.scheduler.submit(run_spec)
            job_ids.append(run_job_id)
            manifest["run_job_id"] = run_job_id
            manifest["state"] = "running"
            _atomic_json(manifest_path, manifest)
            self.event_callback(
                "job_submitted", {"stage": "run", "job_id": run_job_id}
            )
            run_result = self.scheduler.wait(
                run_job_id,
                poll_seconds=self.profile.scheduler.poll_seconds,
                on_state=lambda state: self.event_callback(
                    "job_state", {"stage": "run", "state": state}
                ),
            )
            self._require_success("native Engine run", run_result)
            log_dir = validate_native_artifacts(run_plan, run_id)
            manifest["state"] = "completed"
            _atomic_json(manifest_path, manifest)
            return ClusterRunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                log_dir=log_dir,
                job_ids=tuple(job_ids),
            )
        except BaseException as error:
            manifest["state"] = "failed"
            manifest["error"] = redact_text(str(error))
            manifest["job_ids"] = job_ids
            _atomic_json(manifest_path, manifest)
            raise

    def _compile(self, request: ClusterRunRequest) -> TaskDefinition:
        definition = self.compiler.compile(request.task_dir, request.options)
        updates: dict[str, str] = {}
        if request.model is not None:
            updates["model"] = request.model
        if request.reasoning_effort is not None:
            updates["reasoning_effort"] = request.reasoning_effort
        if updates:
            definition = definition.model_copy(
                update={"agent": definition.agent.model_copy(update=updates)}
            )
        return definition

    def _validate_runtime_inputs(self, request: ClusterRunRequest) -> None:
        if request.agent_name == "codex" and request.model is None:
            raise SetupError("cluster Codex runs require an explicit --model")
        if request.agent_auth is not None and request.agent_auth.value == "local":
            if not (Path.home() / ".codex" / "auth.json").is_file():
                raise SetupError("local Codex authentication is unavailable")
        if shutil.which(request.agent_name) is None:
            raise SetupError(
                f"cluster agent executable is unavailable: {request.agent_name}"
            )
        for label, path in (("Apptainer", self.profile.apptainer.binary),):
            if not path.is_file():
                raise SetupError(f"{label} binary does not exist: {path}")

    def _run_id(self, task_id: str) -> str:
        timestamp = self.clock().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        task = _safe_component(task_id, limit=36)
        return f"{timestamp}-{task}-{secrets.token_hex(4)}"

    def _job_name(self, task_id: str, stage: str) -> str:
        task = _safe_component(task_id, limit=36)
        owner = _safe_component(self.profile.owner, limit=20)
        return f"mt-rsi-{task}-{stage}-{owner}"

    def _build_spec(
        self,
        run_dir: Path,
        name: str,
        resources: ClusterResources,
        script: Path,
    ) -> LSFJobSpec:
        return LSFJobSpec(
            name=name,
            queue=self.profile.scheduler.queue,
            group=self.profile.scheduler.group,
            cpu_slots=self.profile.builder.cpu_slots,
            memory_mb=self.profile.builder.memory_mb,
            walltime=resources.build_walltime,
            stdout_path=(run_dir / "build" / "lsf.%J.out").resolve(),
            stderr_path=(run_dir / "build" / "lsf.%J.err").resolve(),
            script_path=script.resolve(),
            local_tmp_mb=max(
                self.profile.builder.min_tmp_mb,
                resources.local_tmp_mb,
            ),
        )

    def _run_spec(
        self,
        run_dir: Path,
        name: str,
        resources: ClusterResources,
        script: Path,
    ) -> LSFJobSpec:
        return LSFJobSpec(
            name=name,
            queue=self.profile.scheduler.queue,
            group=self.profile.scheduler.group,
            cpu_slots=resources.cpu_slots,
            memory_mb=resources.memory_mb,
            walltime=resources.run_walltime,
            stdout_path=(run_dir / "run" / "lsf.%J.out").resolve(),
            stderr_path=(run_dir / "run" / "lsf.%J.err").resolve(),
            script_path=script.resolve(),
            gpu_count=resources.total_gpus,
            local_tmp_mb=resources.local_tmp_mb,
        )

    def _freeze_control(self, task_dir: Path, control: Path) -> tuple[Path, Path]:
        frozen_task = control / "task"
        shutil.copytree(task_dir, frozen_task, symlinks=True)
        source = self.source_root / "src"
        if not source.is_dir():
            raise SetupError(f"RSI-Harness source tree is unavailable: {source}")
        frozen_source = control / "engine-source"
        shutil.copytree(
            source,
            frozen_source / "src",
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        (control / "profile.json").write_text(
            self.profile.model_dump_json(indent=2)
        )
        _atomic_json(control / "source.json", self._source_metadata())
        return frozen_task.resolve(), frozen_source.resolve()

    def _source_metadata(self) -> dict[str, object]:
        def git(*args: str) -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                ("git", "-C", str(self.source_root), *args),
                check=False,
                capture_output=True,
            )

        commit = git("rev-parse", "HEAD")
        diff = git("diff", "--binary", "HEAD")
        return {
            "commit": commit.stdout.decode(errors="replace").strip()
            if commit.returncode == 0
            else None,
            "dirty": bool(diff.stdout),
            "diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        }

    @staticmethod
    def _relocate_definition(
        definition: TaskDefinition,
        frozen_task: Path,
    ) -> TaskDefinition:
        build_context = frozen_task / "environment"
        service = definition.service.model_copy(
            update={"build_context": build_context}
        )
        return definition.model_copy(
            update={"source_dir": frozen_task, "service": service}
        )

    def _run_plan(
        self,
        definition: TaskDefinition,
        image: SIFImagePlan,
        resources: ClusterResources,
        run_dir: Path,
    ) -> RunPlan:
        devices = tuple(
            GPUDevice(index=index, uuid=f"LSF-{index}", name="LSF allocated GPU")
            for index in range(resources.total_gpus)
        )
        work = GPUAllocation(devices=devices[: resources.work_gpus])
        verifier = GPUAllocation(devices=devices[resources.work_gpus :])
        mode = (
            JudgeGPUMode.DISJOINT
            if verifier.devices
            else JudgeGPUMode.FREEZE_ONLY
        )
        workdir = (
            definition.workdir
            or definition.service.workdir
            or self.profile.apptainer.workspace_target
        )
        if workdir == PurePosixPath("/"):
            workdir = self.profile.apptainer.workspace_target
        runtime_image = ImagePlan(
            base_ref=str(image.sif_path),
            work_ref=str(image.sif_path),
            judge_ref=str(image.sif_path),
            workdir=workdir,
            rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            base_digest=f"sha256:{image.cache_key}",
            work_digest=f"sha256:{image.cache_key}",
            judge_digest=f"sha256:{image.cache_key}",
        )
        return self.compiler.finalize(
            definition,
            runtime_image,
            RunGPUPlan(
                authorized_pool=GPUAllocation(devices=devices),
                work=work,
                judge=verifier,
                judge_mode=mode,
            ),
            RunPaths(
                root=run_dir.resolve(),
                workspace=(run_dir / "workspace").resolve(),
                logs=self.profile.storage.logs_root.resolve(),
            ),
            "bluevela-apptainer-engine",
        )

    def _manifest(
        self,
        *,
        run_id: str,
        run_dir: Path,
        definition: TaskDefinition,
        resources: ClusterResources,
        image: SIFImagePlan,
        agent_version: str | None,
    ) -> dict[str, object]:
        leaf = (
            self.profile.storage.logs_root
            / "runs"
            / run_id
            / definition.task_id
        )
        return {
            "run_id": run_id,
            "owner": self.profile.owner,
            "purpose": f"RSI Harness task {definition.task_id}",
            "submitted_at_utc": self.clock().astimezone(UTC).isoformat(),
            "run_dir": str(run_dir),
            "build_job_id": None,
            "run_job_id": None,
            "source": self._source_metadata(),
            "sif_path": str(image.sif_path),
            "sif_sha256": None,
            "resources": resources.model_dump(mode="json"),
            "agent_version": agent_version,
            "expected_outputs": [
                str(leaf / "final_result.json"),
                str(leaf / "agent_output.txt"),
                str(leaf / "run_agent.log"),
                str(leaf / "submissions" / "agent-1" / "report.json"),
            ],
            "state": "prepared",
        }

    @staticmethod
    def _control_hashes(control: Path) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path in sorted(control.rglob("*")):
            if path.is_symlink():
                content = os.readlink(path).encode()
            elif path.is_file():
                content = path.read_bytes()
            else:
                continue
            hashes[path.relative_to(control).as_posix()] = hashlib.sha256(
                content
            ).hexdigest()
        return hashes

    @staticmethod
    def _require_success(stage: str, result: LSFJobResult) -> None:
        if result.state != "DONE" or result.exit_code != 0:
            raise InfrastructureError(
                f"{stage} job {result.job_id} ended in {result.state} "
                f"with exit {result.exit_code}"
            )


def build_cluster_adapter(
    name_or_path: str | Path,
    *,
    event_callback: Callable[[str, object], None] | None = None,
) -> ClusterAdapter:
    profile = load_cluster_profile(name_or_path)
    if profile.adapter != "bluevela":
        raise SetupError(f"unsupported cluster adapter: {profile.adapter}")
    return BlueVelaClusterAdapter(profile, event_callback=event_callback)
