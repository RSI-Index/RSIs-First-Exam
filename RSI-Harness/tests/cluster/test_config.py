from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError

PROFILE_ENV = {
    "USER": "alice",
    "RSI_CLUSTER_ROOT": "/shared/rsi",
    "RSI_LSF_GROUP": "test-group",
}


def test_packaged_lsf_apptainer_profile_has_no_implicit_site_data_access() -> None:
    profile = load_cluster_profile("lsf-apptainer", PROFILE_ENV)

    assert profile.name == profile.adapter == "lsf-apptainer"
    assert profile.owner == "alice"
    assert profile.scheduler.kind == "lsf"
    assert profile.scheduler.queue == "normal"
    assert profile.scheduler.exclusive is True
    assert profile.scheduler.group == "test-group"
    assert profile.scheduler.remote_binary == "blaunch"
    assert profile.scheduler.remote_host_flag == "-z"
    assert profile.storage.run_root == Path("/shared/rsi/runs")
    assert profile.storage.image_cache == Path("/shared/rsi/images")
    assert profile.storage.hf_home == Path("/shared/rsi/hf-home")
    assert profile.builder.cpu_slots == 64
    assert profile.builder.memory_mb == 524288
    assert profile.builder.min_tmp_mb == 71680
    assert profile.builder.rootless_apt_sandbox is True
    assert profile.builder.faked_binary is None
    assert profile.builder.fakeroot_library is None
    assert profile.apptainer.temp_root == Path("/tmp")
    assert profile.apptainer.environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.work_environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.judge_environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.extra_binds == ()
    assert profile.apptainer.work_binds == ()
    assert profile.apptainer.judge_binds == ()
    assert profile.apptainer.judge_authority_root == Path("/shared/rsi/authority")
    assert profile.resources.gpus_per_node == 8


def _packaged_text() -> str:
    return resources.files("rsi_harness.cluster.lsf_apptainer").joinpath(
        "profile.toml"
    ).read_text()


def test_external_profile_overrides_tools_and_phase_scoped_data(tmp_path) -> None:
    profile_path = tmp_path / "site.toml"
    profile_path.write_text(
        _packaged_text()
        .replace('name = "lsf-apptainer"', 'name = "research-cluster"')
        .replace('binary = "/usr/bin/apptainer"', 'binary = "${TOOLS}/apptainer"')
        .replace('binary = "/usr/bin/podman"', 'binary = "${TOOLS}/podman"')
        .replace('work_binds = []', '''work_binds = [
  { source = "${DATA_ROOT}/train", target = "/rsi-data/train", read_only = true },
]''')
        .replace('judge_binds = []', '''judge_binds = [
  { source = "${DATA_ROOT}/eval", target = "/rsi-data/eval", read_only = true },
]''')
    )
    profile = load_cluster_profile(profile_path, {
        **PROFILE_ENV, "TOOLS": "/opt/site/bin", "DATA_ROOT": "/shared/data",
    })

    assert profile.name == "research-cluster"
    assert profile.adapter == "lsf-apptainer"
    assert profile.apptainer.binary == Path("/opt/site/bin/apptainer")
    assert profile.builder.binary == Path("/opt/site/bin/podman")
    assert profile.apptainer.work_binds[0].source == Path("/shared/data/train")
    assert profile.apptainer.judge_binds[0].source == Path("/shared/data/eval")
    assert all(binding.read_only for binding in profile.apptainer.work_binds)
    assert all(binding.read_only for binding in profile.apptainer.judge_binds)
    assert profile.apptainer.extra_binds == ()


def test_profile_expands_only_explicit_environment_references(tmp_path) -> None:
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(_packaged_text().replace("${RSI_CLUSTER_ROOT}", "${RUN_BASE}"))
    profile = load_cluster_profile(profile_path, {**PROFILE_ENV, "RUN_BASE": "/x"})
    assert profile.storage.run_root == Path("/x/runs")


@pytest.mark.parametrize("missing", ["USER", "RSI_CLUSTER_ROOT", "RSI_LSF_GROUP"])
def test_packaged_profile_requires_explicit_site_environment(missing) -> None:
    environ = dict(PROFILE_ENV)
    del environ[missing]
    with pytest.raises(SetupError, match=missing):
        load_cluster_profile("lsf-apptainer", environ)


def test_profile_rejects_relative_shared_storage() -> None:
    with pytest.raises(SetupError, match="absolute"):
        load_cluster_profile("lsf-apptainer", {**PROFILE_ENV, "RSI_CLUSTER_ROOT": "relative"})


def test_profile_rejects_unknown_environment_reference(tmp_path) -> None:
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(_packaged_text().replace("${RSI_CLUSTER_ROOT}", "${MISSING}"))
    with pytest.raises(SetupError, match="MISSING"):
        load_cluster_profile(profile_path, PROFILE_ENV)


def test_profile_rejects_unknown_cluster_name() -> None:
    with pytest.raises(SetupError, match="unknown cluster"):
        load_cluster_profile("not-a-cluster", PROFILE_ENV)


@pytest.mark.parametrize("empty", ["USER", "RSI_CLUSTER_ROOT", "RSI_LSF_GROUP"])
def test_packaged_profile_rejects_empty_site_environment(empty) -> None:
    with pytest.raises(SetupError, match=empty):
        load_cluster_profile("lsf-apptainer", {**PROFILE_ENV, empty: ""})
