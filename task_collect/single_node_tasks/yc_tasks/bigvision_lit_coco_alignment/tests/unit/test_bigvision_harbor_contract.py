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
        "environment/project/candidate/text_tower.py",
        "environment/project/candidate/alignment_loss.py",
        "environment/project/candidate/config.toml",
        "environment/task_tools/lit_coco_autoresearch.py",
        "environment/task_tools/model_adapter.py",
        "environment/task_tools/contrastive_autoresearch.py",
        "environment/task_tools/run_candidate.py",
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
    assert task["task"]["name"] == "scale-autoresearch/bigvision-lit-coco-alignment"
    assert task["metadata"]["baseline"]["score_status"] == "complete"
    assert task["agent"]["network_mode"] == "none"
    assert task["environment"]["network_mode"] == "none"
    assert task["verifier"]["environment_mode"] == "separate"
    assert task["verifier"]["network_mode"] == "none"
    assert task["environment"]["gpus"] == 8
    assert task["verifier"]["environment"]["gpus"] == 8


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
    assert root_contract["reference_metric"] == pytest.approx(38.9245)
    assert root_contract["imagenet_retention_lower_bound"] == pytest.approx(19.665)
    assert root_contract["trainable_parameter_ceiling"] == 111_000_000
    assert root_contract["training_flop_ceiling"] == pytest.approx(2.0e18)
    assert root_contract["calibration_provenance"]["trainable_parameter_ceiling"]["evidence_class"] == "engineering_estimate"


def test_npz_validator_rejects_object_arrays(tmp_path: Path) -> None:
    checker = load_module("bv_artifact", "tests/artifact_check.py")
    good = tmp_path / "good.npz"
    np.savez(good, **{"txt/kernel": np.ones((2, 2), dtype=np.float32)})
    record = checker.inspect_npz(good)
    assert record["array_count"] == 1

    bad = tmp_path / "bad.npz"
    np.savez(bad, payload=np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError, match="object"):
        checker.inspect_npz(bad)


def test_score_fails_closed_for_incomplete_contract() -> None:
    score = load_module("bv_score", "tests/score.py")
    metrics = {"status": "pass", "i2t_r1": 50.0, "t2i_r1": 40.0, "imagenet_zero_shot_accuracy": 22.0}
    with pytest.raises(score.BaselineContractError):
        score.compute_reward(metrics, {"status": "incomplete"})


def test_score_enforces_retention_and_geometric_mean() -> None:
    score = load_module("bv_score_ready", "tests/score.py")
    baseline = {
        "status": "complete",
        "reference_metric": 39.0,
        "imagenet_retention_lower_bound": 20.0,
        "trainable_parameter_ceiling": 110_000_000,
        "training_flop_ceiling": 1.0e18,
    }
    result = score.compute_reward(
        {"status": "pass", "i2t_r1": 49.0, "t2i_r1": 36.0, "imagenet_zero_shot_accuracy": 21.0},
        baseline,
    )
    assert result["primary_metric"] == pytest.approx(42.0)
    assert result["reward"] == pytest.approx(0.42)
    assert result["retention_gate"] == "pass"

    with pytest.raises(ValueError, match="retention"):
        score.compute_reward(
            {"status": "pass", "i2t_r1": 49.0, "t2i_r1": 36.0, "imagenet_zero_shot_accuracy": 19.9},
            baseline,
        )


def test_metric_parser_requires_all_clean_metrics(tmp_path: Path) -> None:
    evaluate = load_module("bv_evaluate_parser", "tests/evaluate.py")
    raw = {"step": 0, "z/0shot/imagenet2012_accuracy": 0.21}
    for index, key in enumerate(evaluate.RETRIEVAL_KEYS.values(), 1):
        raw[key] = index / 10
    path = tmp_path / "big_vision_metrics.txt"
    path.write_text(json.dumps(raw) + "\n")
    parsed = evaluate.parse_metrics(path)
    assert parsed["status"] == "pass"
    assert parsed["i2t_r1"] == pytest.approx(10.0)
    assert parsed["imagenet_zero_shot_accuracy"] == pytest.approx(21.0)
    del raw["z/retr/coco_txt2img_recall@1"]
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="missing metrics"):
        evaluate.parse_metrics(path)


def test_policy_gate_allows_only_three_files_and_rejects_eval_import(tmp_path: Path) -> None:
    checker = load_module("bv_policy", "tests/policy_check.py")
    import yaml

    clean = tmp_path / "clean"
    candidate = tmp_path / "candidate"
    shutil.copytree(ROOT / "environment" / "project", clean)
    shutil.copytree(clean, candidate)
    (candidate / "candidate" / "text_tower.py").write_text(
        "from big_vision.evaluators import common\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    provenance = {field: "x" for field in yaml.safe_load((ROOT / "tests/policy.yaml").read_text())["provenance"]["required_fields"]}
    provenance["web_search"] = "disabled"
    (output / "provenance.json").write_text(json.dumps(provenance))
    report = checker.check(yaml.safe_load((ROOT / "tests/policy.yaml").read_text()), clean, candidate, output)
    assert report["decision"] == "FAIL"
    assert any("forbidden import" in item["reason"] for item in report["violations"])
