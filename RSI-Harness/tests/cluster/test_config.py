from __future__ import annotations

from importlib import resources

import pytest

from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError


def test_packaged_bluevela_profile_contains_cluster_policy() -> None:
    profile = load_cluster_profile("bluevela", {"USER": "alice"})

    assert profile.name == "bluevela"
    assert profile.owner == "alice"
    assert profile.scheduler.kind == "lsf"
    assert profile.scheduler.queue == "normal"
    assert profile.scheduler.group == "grp_models"
    assert profile.storage.run_root.is_absolute()
    assert profile.builder.cpu_slots == 64
    assert profile.builder.memory_mb == 524288
    assert profile.builder.min_tmp_mb == 71680
    assert profile.builder.rootless_apt_sandbox is True
    assert profile.builder.faked_binary is not None
    assert profile.builder.fakeroot_library is not None
    assert profile.apptainer.temp_root.as_posix() == "/tmp"
    assert profile.resources.gpus_per_node == 8


def test_profile_expands_only_explicit_environment_references(tmp_path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            "/proj/datasets/interns/yuetai/agent_envs/more_task/rsi-harness",
            "${RUN_BASE}/rsi-harness",
        )
    )

    profile = load_cluster_profile(profile_path, {"USER": "alice", "RUN_BASE": "/x"})

    assert profile.storage.run_root.as_posix() == "/x/rsi-harness/runs"


def test_profile_rejects_unknown_environment_reference(tmp_path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            "/proj/datasets/interns/yuetai/agent_envs/more_task/rsi-harness",
            "${MISSING}/runs",
        )
    )

    with pytest.raises(SetupError, match="MISSING"):
        load_cluster_profile(profile_path, {"USER": "alice"})


def test_profile_rejects_unknown_cluster_name() -> None:
    with pytest.raises(SetupError, match="unknown cluster"):
        load_cluster_profile("not-a-cluster", {"USER": "alice"})
