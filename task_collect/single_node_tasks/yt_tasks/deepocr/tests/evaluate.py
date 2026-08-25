#!/usr/bin/env python3
from __future__ import annotations

import json
import gc
import importlib.util
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def find_dataset() -> tuple[Path, Path]:
    configured = os.getenv("OMNIDOCBENCH_DIR")
    root = Path(configured) if configured else Path(snapshot_download(repo_id="opendatalab/OmniDocBench", repo_type="dataset"))
    image_candidates = [path for path in root.rglob("images") if path.is_dir()]
    gt_candidates = [path for path in root.rglob("*.json") if "omnidocbench" in path.name.lower() and "demo" not in path.name.lower()]
    if not image_candidates or not gt_candidates:
        raise RuntimeError(f"could not locate OmniDocBench images/ground truth below {root}")
    return image_candidates[0], sorted(gt_candidates, key=lambda path: path.stat().st_size, reverse=True)[0]


def metric_value(payload: dict, *path: str) -> float:
    value = payload
    for part in path:
        value = value[part]
    return float(value)


def run_official_inference(model: Path, images: Path, predictions: Path) -> list[int]:
    """Run upstream inference while measuring the actual inserted media sequence."""
    sys.path.insert(0, "/opt/project")
    import llava

    visual_token_counts: list[int] = []
    loaded_models = []
    original_load = llava.load

    def trusted_load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        loaded_models.append(loaded)
        original_outer_generate = loaded.generate
        original_llm_generate = loaded.llm.generate
        pending: list[tuple[int, int]] = []

        def outer_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise RuntimeError("trusted visual-token meter requires input_ids")
            image_token_id = loaded.tokenizer.media_token_ids["image"]
            placeholders = int((input_ids == image_token_id).sum().item())
            pending.append((int(input_ids.shape[1]), placeholders))
            return original_outer_generate(*args, **kwargs)

        def llm_generate(*args, **kwargs):
            if not pending:
                raise RuntimeError("LLM generation escaped the trusted media-token meter")
            text_tokens, placeholders = pending.pop()
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None:
                raise RuntimeError("clean DeepOCR path did not provide inputs_embeds")
            inserted = int(inputs_embeds.shape[1]) - text_tokens + placeholders
            if inserted <= 0:
                raise RuntimeError(f"invalid measured visual-token count: {inserted}")
            visual_token_counts.append(inserted)
            return original_llm_generate(*args, **kwargs)

        loaded.generate = outer_generate
        loaded.llm.generate = llm_generate
        return loaded

    llava.load = trusted_load
    spec = importlib.util.spec_from_file_location(
        "trusted_deepocr_official", "/opt/project/llava/eval/omini_doc_bench.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned DeepOCR evaluator")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    old_argv = sys.argv
    sys.argv = [
        "omini_doc_bench.py",
        "--model-path",
        str(model),
        "--input-folder",
        str(images),
        "--output-folder",
        str(predictions),
        "--text",
        "Free OCR.",
    ]
    try:
        with Path("/logs/verifier/deepocr_inference.log").open("w") as log, \
             contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            official.main()
    finally:
        sys.argv = old_argv
        llava.load = original_load
        loaded_models.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return visual_token_counts


def main() -> None:
    model = Path("/app/output/model")
    if not (model / "config.json").is_file():
        raise RuntimeError("missing complete Hugging Face checkpoint at /app/output/model")
    model_config = json.loads((model / "config.json").read_text())
    llm_config = model_config.get("llm_cfg") or {}
    if (
        llm_config.get("model_type") != "qwen2"
        or int(llm_config.get("hidden_size", 0)) != 3584
        or int(llm_config.get("num_hidden_layers", 0)) != 28
    ):
        raise RuntimeError("the fixed decoder family must remain Qwen2-VL-7B")
    images, ground_truth = find_dataset()
    work = Path("/tmp/deepocr_official_eval")
    predictions = work / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    measured_visual_tokens = run_official_inference(model, images, predictions)
    expected_images = sum(
        1 for path in images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    )
    prediction_count = len(list(predictions.glob("*.md")))
    if prediction_count != expected_images or len(measured_visual_tokens) != expected_images:
        raise RuntimeError(
            f"incomplete DeepOCR output/token telemetry: images={expected_images} "
            f"predictions={prediction_count} token_records={len(measured_visual_tokens)}"
        )

    config = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist"]},
                "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 8},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(ground_truth)},
                "prediction": {"data_path": str(predictions)},
                "match_method": "quick_match",
                "match_workers": 8,
            },
        }
    }
    config_path = work / "deepocr.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (work / "result").mkdir(exist_ok=True)
    scored = subprocess.run(
        ["/opt/omnidoc-venv/bin/python", "/opt/OmniDocBench/pdf_validation.py", "--config", str(config_path)],
        cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "/opt/OmniDocBench"},
    )
    Path("/logs/verifier/omnidocbench.log").write_text(scored.stdout)
    if scored.returncode:
        raise RuntimeError(f"OmniDocBench official evaluator exited {scored.returncode}")
    result_files = sorted((work / "result").glob("*_metric_result.json"), key=lambda path: path.stat().st_mtime)
    if not result_files:
        raise RuntimeError("OmniDocBench did not emit metric JSON")
    raw = json.loads(result_files[-1].read_text())
    provenance = json.loads(Path("/app/output/provenance.json").read_text())
    average_visual_tokens = sum(measured_visual_tokens) / len(measured_visual_tokens)
    declared_visual_tokens = float(provenance["average_visual_tokens"])
    if abs(declared_visual_tokens - average_visual_tokens) > max(1.0, average_visual_tokens * 0.01):
        raise RuntimeError(
            f"declared average_visual_tokens={declared_visual_tokens} does not match "
            f"trusted measurement={average_visual_tokens}"
        )
    metrics = {
        "text_edit_distance": metric_value(raw, "text_block", "all", "Edit_dist", "ALL_page_avg"),
        "formula_edit_distance": metric_value(raw, "display_formula", "all", "Edit_dist", "ALL_page_avg"),
        "table_teds": metric_value(raw, "table", "page", "TEDS", "ALL"),
        "reading_order_edit_distance": metric_value(raw, "reading_order", "all", "Edit_dist", "ALL_page_avg"),
        "average_visual_tokens": average_visual_tokens,
        "evaluated_pages": expected_images,
    }
    Path("/logs/verifier/metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
