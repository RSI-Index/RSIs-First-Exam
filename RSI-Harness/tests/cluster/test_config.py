from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError


def test_packaged_bluevela_profile_contains_cluster_policy() -> None:
    profile = load_cluster_profile("bluevela", {"USER": "alice"})

    assert profile.name == "bluevela"
    assert profile.owner == "alice"
    assert profile.scheduler.kind == "lsf"
    assert profile.scheduler.queue == "normal"
    assert profile.scheduler.exclusive is True
    assert profile.scheduler.group == "grp_models"
    assert profile.scheduler.remote_binary == "blaunch"
    assert profile.scheduler.remote_host_flag == "-z"
    assert profile.storage.run_root.is_absolute()
    assert profile.builder.cpu_slots == 64
    assert profile.builder.memory_mb == 524288
    assert profile.builder.min_tmp_mb == 71680
    assert profile.builder.rootless_apt_sandbox is True
    assert profile.builder.faked_binary is not None
    assert profile.builder.fakeroot_library is not None
    assert profile.apptainer.temp_root.as_posix() == "/tmp"
    assert profile.apptainer.environment["RSI_SHARED_DATA_ROOT"].startswith("/proj/")
    assert profile.apptainer.work_environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.judge_environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.extra_binds == (Path("/proj"),)
    assert profile.apptainer.work_binds
    assert profile.apptainer.judge_binds
    assert not any(
        "paloma" in str(binding.source)
        for binding in profile.apptainer.work_binds
    )
    assert any(
        "paloma" in str(binding.source)
        for binding in profile.apptainer.judge_binds
    )
    assert all(binding.read_only for binding in profile.apptainer.work_binds)
    assert all(binding.read_only for binding in profile.apptainer.judge_binds)
    assert profile.resources.gpus_per_node == 8


def test_profile_runtime_environment_is_expanded_and_portable(tmp_path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            "RSI_SHARED_DATA_ROOT = "
            '"/proj/datasets/interns/yuetai/agent_envs/more_task"',
            "RSI_SHARED_DATA_ROOT = \"${DATA_ROOT}\"",
        )
    )

    profile = load_cluster_profile(
        profile_path,
        {"USER": "alice", "DATA_ROOT": "/cluster/shared/rsi-data"},
    )

    assert profile.apptainer.environment == {
        "RSI_SHARED_DATA_ROOT": "/cluster/shared/rsi-data"
    }


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
