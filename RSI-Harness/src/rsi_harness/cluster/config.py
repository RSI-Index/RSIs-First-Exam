"""Strict, portable cluster profile loading."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from rsi_harness.errors import SetupError

_ENV_REFERENCE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class _ProfileModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SchedulerProfile(_ProfileModel):
    kind: Literal["lsf"]
    submit_binary: str = "bsub"
    status_binary: str = "bjobs"
    cancel_binary: str = "bkill"
    queue: str
    group: str
    poll_seconds: float = Field(default=10.0, gt=0.0)


class StorageProfile(_ProfileModel):
    run_root: Path
    image_cache: Path
    logs_root: Path
    hf_home: Path
    hf_datasets_cache: Path

    @field_validator(
        "run_root", "image_cache", "logs_root", "hf_home", "hf_datasets_cache"
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("cluster storage paths must be absolute")
        return value


class BuilderProfile(_ProfileModel):
    binary: Path
    cpu_slots: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    walltime: str
    temp_root: Path = Path("/tmp")
    min_tmp_mb: int = Field(gt=0)
    rootless_apt_sandbox: bool = False
    faked_binary: Path | None = None
    fakeroot_library: Path | None = None

    @model_validator(mode="after")
    def _paired_fakeroot_tools(self) -> Self:
        if (self.faked_binary is None) != (self.fakeroot_library is None):
            raise ValueError(
                "builder faked_binary and fakeroot_library must be set together"
            )
        return self


class ApptainerProfile(_ProfileModel):
    binary: Path
    dns_bind: Path = Path("/etc/resolv.conf")
    extra_binds: tuple[Path, ...] = ()
    build_args: tuple[str, ...] = ()
    workspace_target: PurePosixPath = PurePosixPath("/testbed")
    temp_root: Path = Path("/tmp")

    @field_validator("workspace_target")
    @classmethod
    def _workspace_target(cls, value: PurePosixPath) -> PurePosixPath:
        if (
            not value.is_absolute()
            or value == PurePosixPath("/")
            or ".." in value.parts
        ):
            raise ValueError(
                "Apptainer workspace target must be an absolute non-root path"
            )
        return value


class ResourceProfile(_ProfileModel):
    min_cpu_slots: int = Field(gt=0)
    min_memory_mb: int = Field(gt=0)
    gpus_per_node: int = Field(gt=0)
    walltime_margin_seconds: int = Field(ge=0)
    all_gpus_override: int | None = Field(default=None, gt=0)


class ClusterProfile(_ProfileModel):
    name: str
    adapter: Literal["bluevela"]
    owner: str
    scheduler: SchedulerProfile
    storage: StorageProfile
    builder: BuilderProfile
    apptainer: ApptainerProfile
    resources: ResourceProfile


def _expand(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in environ:
                raise SetupError(
                    f"cluster profile references missing environment variable {name}"
                )
            return environ[name]

        return _ENV_REFERENCE.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, environ) for key, item in value.items()}
    return value


def _profile_bytes(name_or_path: str | Path) -> bytes:
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.read_bytes()
    if str(name_or_path) == "bluevela":
        return (
            resources.files("rsi_harness.cluster.bluevela")
            .joinpath("profile.toml")
            .read_bytes()
        )
    raise SetupError(f"unknown cluster or profile path: {name_or_path}")


def load_cluster_profile(
    name_or_path: str | Path,
    environ: Mapping[str, str] | None = None,
) -> ClusterProfile:
    """Load a packaged cluster name or an explicit TOML profile."""
    try:
        raw = tomllib.loads(_profile_bytes(name_or_path).decode("utf-8"))
        expanded = _expand(raw, os.environ if environ is None else environ)
        return ClusterProfile.model_validate(expanded)
    except SetupError:
        raise
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise SetupError(f"invalid cluster profile {name_or_path}: {error}") from error
