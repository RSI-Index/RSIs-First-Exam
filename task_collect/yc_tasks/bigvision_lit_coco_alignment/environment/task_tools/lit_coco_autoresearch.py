"""Closed-schema adapter around the pinned historical LiT/COCO config."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from big_vision.configs.proj.image_text import lit_coco
from ml_collections import ConfigDict


CONFIG_PATH = Path(os.environ.get("BIGVISION_CANDIDATE_CONFIG", "/app/project/candidate/config.toml"))
IMAGE_INIT = os.environ.get("BIGVISION_IMAGE_INIT", "/datasets/initializers/vit_b16_augreg.npz")
TEXT_INIT = os.environ.get("BIGVISION_TEXT_INIT", "/datasets/initializers/bert_base")
EVAL_CHECKPOINT = os.environ.get("BIGVISION_EVAL_CHECKPOINT")


def _closed_table(value: dict, name: str, keys: set[str]) -> None:
    extra = set(value) - keys
    if extra:
        raise ValueError(f"unsupported candidate {name} keys: {sorted(extra)}")


def load_candidate_config(path: Path = CONFIG_PATH) -> dict:
    value = tomllib.loads(path.read_text())
    _closed_table(value, "top-level", {"schema_version", "text", "model", "loss"})
    if value.get("schema_version") != 1:
        raise ValueError("candidate config schema_version must equal 1")
    for table in ("text", "model", "loss"):
        if not isinstance(value.get(table), dict):
            raise ValueError(f"candidate config requires [{table}]")
    _closed_table(value["text"], "text", {"config", "head_zeroinit"})
    _closed_table(value["model"], "model", {"temperature_init"})
    _closed_table(value["loss"], "loss", {"name", "label_smoothing"})
    if value["text"].get("config") not in {"base", "large"}:
        raise ValueError("text.config must be base or large")
    if not isinstance(value["text"].get("head_zeroinit"), bool):
        raise ValueError("text.head_zeroinit must be boolean")
    temperature = value["model"].get("temperature_init")
    if not isinstance(temperature, (int, float)) or not 0.01 <= float(temperature) <= 100.0:
        raise ValueError("model.temperature_init must be in [0.01,100]")
    if value["loss"].get("name") != "bidirectional_contrastive":
        raise ValueError("loss.name must preserve the fixed trainer interface")
    smoothing = value["loss"].get("label_smoothing")
    if not isinstance(smoothing, (int, float)) or not 0.0 <= float(smoothing) <= 0.2:
        raise ValueError("loss.label_smoothing must be in [0,0.2]")
    return value


def _replace_vocab(config) -> None:
    local_vocab = f"{TEXT_INIT}/vocab.txt"
    for path in (("input", "pp"), ("evals", "val", "pp_fn"), ("evals", "coco", "pp_fn"),
                 ("evals", "imagenet", "pp_fn"), ("evals", "disclf", "pp_txt"),
                 ("evals", "retrieval_coco", "pp_txt")):
        owner = config
        try:
            for key in path[:-1]:
                owner = owner[key]
            owner[path[-1]] = owner[path[-1]].replace(
                "gs://vit_models/lit/bert/uncased_L-12_H-768_A-12/vocab.txt", local_vocab)
        except (KeyError, TypeError):
            continue


def _assert_outer_contract(config) -> None:
    expected = {
        "total_steps": 5000,
        "optax_name": "scale_by_adam",
        "lr": 0.001,
        "wd": 0.01,
        "grad_clip_norm": 1.0,
        "loss_use_global_batch": True,
        "seed": 0,
    }
    for key, wanted in expected.items():
        if config[key] != wanted:
            raise AssertionError(f"immutable config mismatch for {key}: {config[key]!r}")
    if config.input.batch_size != 4096 or dict(config.input.data) != {"name": "coco_captions", "split": "train"}:
        raise AssertionError("immutable COCO input/batch contract changed")
    if list(config.schedule) != [("img/.*", None), (".*", {"decay_type": "cosine", "warmup_steps": 150})]:
        raise AssertionError("immutable freeze/schedule contract changed")
    if config.model.image_model != "vit" or config.model.image.variant != "B/16":
        raise AssertionError("immutable ViT-B/16 image model changed")
    if config.model.out_dim != (None, 768):
        raise AssertionError("immutable LiT output dimensions changed")


def get_config(arg=None):
    if arg not in (None, ""):
        raise ValueError("this Harbor adapter accepts no command-line config overrides")
    candidate = load_candidate_config()
    config = lit_coco.get_config("res=224,runlocal=False,token_len=16,txt=bert_base,img=B/16,init=,img_head=False")
    _replace_vocab(config)
    config.model.text_model = "harbor_text_adapter"
    config.model.text = ConfigDict(candidate["text"])
    config.model.temperature_init = float(candidate["model"]["temperature_init"])
    config.model_init = {"image": IMAGE_INIT, "text": TEXT_INIT}
    config.model_load = ConfigDict({
        "txt_load_kw": {"dont_load": ["head/kernel", "head/bias"]},
        "img_load_kw": {"dont_load": ["head/kernel", "head/bias"]},
    })
    _assert_outer_contract(config)
    if os.environ.get("BIGVISION_FINAL_EVAL") == "1":
        if not EVAL_CHECKPOINT:
            raise ValueError("BIGVISION_EVAL_CHECKPOINT is required for final evaluation")
        config.model_init = EVAL_CHECKPOINT
        config.model_load = ConfigDict()
        config.evals = ConfigDict({
            "disclf": config.evals.disclf,
            "retrieval_coco": config.evals.retrieval_coco,
        })
    else:
        config.evals = ConfigDict()
    return config

