from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_harbor_surface_is_complete() -> None:
    required = {
        "baseline_contract.json",
        "harbor_manifest.json",
        "run_harbor.sh",
        "setup/Dockerfile",
        "setup/asset_sources.lock.json",
        "setup/prepare_assets.py",
        "setup/prepare_assets.sh",
        "environment/Dockerfile",
        "environment/docker-compose.yaml",
        "environment/asset_contract.json",
        "environment/project/candidate/selector.py",
        "environment/task_tools/run_candidate.py",
        "environment/task_tools/validate_subset.py",
        "environment/run_candidate.sh",
        "tests/Dockerfile",
        "tests/policy.yaml",
        "tests/policy_check.py",
        "tests/artifact_check.py",
        "tests/evaluate.py",
        "tests/score.py",
        "tests/test.sh",
        "tests/baseline_contract.json",
    }
    assert not sorted(path for path in required if not (ROOT / path).is_file())


def test_task_toml_is_harbor_schema_and_offline() -> None:
    task = tomllib.loads((ROOT / "task.toml").read_text())
    assert task["schema_version"] == "1.3"
    assert task["task"]["name"] == "scale-autoresearch/datacomp-s-filter-discovery"
    assert task["metadata"]["baseline"]["score_status"] == "complete"
    assert task["agent"]["network_mode"] == "none"
    assert task["environment"]["network_mode"] == "none"
    assert task["verifier"]["environment_mode"] == "separate"
    assert task["verifier"]["network_mode"] == "none"
    assert task["environment"]["gpus"] == 4
    assert task["verifier"]["environment"]["gpus"] == 4


def test_task_owned_setup_is_outside_offline_runtime() -> None:
    manifest = json.loads((ROOT / "harbor_manifest.json").read_text())
    assets = json.loads((ROOT / "environment/asset_contract.json").read_text())
    assert manifest["asset_setup"] == "setup/prepare_assets.sh"
    assert manifest["runtime_network"] == "none"
    assert assets["provisioning_mode"] == "task_owned_official_setup_then_read_only_mounts"
    assert assets["network_downloader_in_harbor"] is False


def test_complete_demo_baseline_contract_is_explicit_and_identical() -> None:
    root_contract = json.loads((ROOT / "baseline_contract.json").read_text())
    environment_contract = json.loads((ROOT / "environment/baseline_contract.json").read_text())
    test_contract = json.loads((ROOT / "tests/baseline_contract.json").read_text())
    assert root_contract == environment_contract == test_contract
    assert root_contract["status"] == "complete"
    assert root_contract["calibration_profile"] == "repository_anchored_demo_v1"
    assert root_contract["reference_metric"] == pytest.approx(0.1731364262164997)
    assert root_contract["candidate_threshold"] == pytest.approx(0.1731364262164997)
    assert root_contract["reference_run_id"] == "task6-parallel-ulimit-20260803T162507Z"
    assert root_contract["calibration_provenance"]["reference_metric"]["evidence_class"] == "locally_recomputed_released_control"


def test_subset_validator_accepts_only_sorted_unique_in_universe(tmp_path: Path) -> None:
    validator = load_module("dc_validate_subset", "environment/task_tools/validate_subset.py")
    dtype = np.dtype("u8,u8")
    universe = np.array([(1, 2), (2, 3), (9, 4)], dtype=dtype)
    subset = np.array([(1, 2), (9, 4)], dtype=dtype)
    universe_path = tmp_path / "universe.npy"
    subset_path = tmp_path / "subset.npy"
    np.save(universe_path, universe, allow_pickle=False)
    np.save(subset_path, subset, allow_pickle=False)
    record = validator.validate_subset(subset_path, universe_path)
    assert record["status"] == "pass"
    assert record["count"] == 2

    np.save(subset_path, subset[::-1], allow_pickle=False)
    with pytest.raises(ValueError, match="lexicographically"):
        validator.validate_subset(subset_path, universe_path)


def test_score_fails_closed_for_incomplete_contract(tmp_path: Path) -> None:
    score = load_module("dc_score", "tests/score.py")
    metrics = {
        "status": "pass",
        "component_metrics": {f"task_{i:02d}": 0.2 for i in range(38)},
    }
    with pytest.raises(score.BaselineContractError):
        score.compute_reward(metrics, {"status": "incomplete"})


def test_score_uses_exactly_38_finite_components() -> None:
    score = load_module("dc_score_ready", "tests/score.py")
    baseline = {"status": "complete", "reference_metric": 0.17}
    values = {f"task_{i:02d}": 0.1 + i / 1000 for i in range(38)}
    result = score.compute_reward(
        {"status": "pass", "component_metrics": values}, baseline
    )
    assert result["component_count"] == 38
    assert result["reward"] == pytest.approx(sum(values.values()) / 38)


def test_policy_gate_rejects_noneditable_addition_and_network_import(tmp_path: Path) -> None:
    checker = load_module("dc_policy", "tests/policy_check.py")
    import yaml

    clean = tmp_path / "clean"
    candidate = tmp_path / "candidate"
    shutil.copytree(ROOT / "environment" / "project", clean)
    shutil.copytree(clean, candidate)
    selector = candidate / "candidate" / "selector.py"
    selector.write_text(selector.read_text() + "\nimport requests\n")
    (candidate / "extra.py").write_text("x = 1\n")
    output = tmp_path / "output"
    output.mkdir()
    policy = yaml.safe_load((ROOT / "tests/policy.yaml").read_text())
    provenance = {field: "x" for field in policy["provenance"]["required_fields"]}
    provenance["web_search"] = "disabled"
    (output / "provenance.json").write_text(json.dumps(provenance))
    report = checker.check(policy, clean, candidate, output)
    assert report["decision"] == "FAIL"
    assert any(item["path"] == "extra.py" for item in report["violations"])
    assert any("forbidden import" in item["reason"] for item in report["violations"])
