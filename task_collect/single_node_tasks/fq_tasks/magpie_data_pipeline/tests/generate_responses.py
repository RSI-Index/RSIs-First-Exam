#!/usr/bin/env python3
"""Generate matched greedy answers for the frozen Magpie evaluation suite."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Sequence

try:
    from .evaluation_contract import TOTAL_PROMPTS
    from .task_contract import CONTRACT, RUNTIME_PATHS, load_llama3_chat_template
except ImportError:  # /tests is deliberately a flat, root-owned verifier tree.
    from evaluation_contract import TOTAL_PROMPTS
    from task_contract import CONTRACT, RUNTIME_PATHS, load_llama3_chat_template


MANIFEST_PATH = Path("/task-assets/eval_manifest.json")
OUTPUT_ROOT = Path("/app/output/verifier")
CANDIDATE_MODEL = Path("/app/output/verifier/hf_model")
BASELINE_MODEL = Path(RUNTIME_PATHS["baseline_checkpoint"])
GENERATION = {"temperature": 0.0, "max_tokens": CONTRACT.generation_output_tokens}
_SOURCES = ("alpaca_eval_2", "arena_hard", "wildbench")


def _ordinary_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"model must be an ordinary directory: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    if not files:
        raise ValueError(f"model directory is empty: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _tokenizer_hash(path: Path) -> str:
    """Fingerprint tokenizer inputs rather than treating model weights as tokenizer state."""

    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
    digest = hashlib.sha256()
    found = False
    for name in names:
        item = path / name
        if item.is_file() and not item.is_symlink():
            found = True
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(item).encode("ascii"))
            digest.update(b"\n")
    if not found:
        raise ValueError("model tokenizer files are missing")
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    _atomic_bytes(
        path,
        b"".join(
            (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            for row in rows
        ),
    )


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, object]]:
    """Read the root-owned selection and retain its deterministic source order."""

    path = Path(path)
    if not _ordinary_file(path):
        raise ValueError("eval manifest must be an ordinary file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, dict) or set(sources) != set(_SOURCES):
        raise ValueError("eval manifest must contain exactly the three evaluation sources")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in _SOURCES:
        entries = sources[source]
        if not isinstance(entries, list) or len(entries) != CONTRACT.eval_per_source:
            raise ValueError("eval manifest must contain 256 entries per source")
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("source") != source:
                raise ValueError("eval manifest entry has invalid source")
            identifier = entry.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("eval manifest prompt ids must be unique")
            if source == "wildbench":
                messages = entry.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("wildbench entry must retain messages")
            elif not isinstance(entry.get("prompt"), str):
                raise ValueError("single-turn entry must contain a prompt")
            seen.add(identifier)
            records.append(entry)
    if len(records) != TOTAL_PROMPTS:
        raise ValueError("eval manifest must contain exactly 768 prompts")
    return records


def format_prompt(tokenizer: Any, entry: dict[str, object], *, chat_template: str) -> str:
    """Use the Llama 3 chat template for both ordinary and WildBench prompts."""

    if entry.get("source") == "wildbench":
        messages = entry.get("messages")
        if not isinstance(messages, list):
            raise ValueError("wildbench messages are required")
    else:
        prompt = entry.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt is required")
        messages = [{"role": "user", "content": prompt}]
    converted: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str):
            raise ValueError("manifest message is invalid")
        converted.append({"role": str(message["role"]), "content": str(message["content"])})
    rendered = tokenizer.apply_chat_template(converted, chat_template=chat_template, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str):
        raise ValueError("Llama tokenizer chat template must render text")
    return rendered


def validate_response_coverage(rows: Iterable[object], expected_ids: Sequence[str]) -> list[dict[str, object]]:
    """Require one non-empty answer for every selected ID, in canonical order."""

    expected = list(expected_ids)
    if len(expected) != len(set(expected)) or len(expected) != TOTAL_PROMPTS:
        raise ValueError("expected response IDs must be exactly 768 unique IDs")
    found: dict[str, dict[str, object]] = {}
    for item in rows:
        if not isinstance(item, dict) or set(item) - {"id", "source", "response"}:
            raise ValueError("response row has an invalid schema")
        identifier, response = item.get("id"), item.get("response")
        if not isinstance(identifier, str) or not isinstance(response, str) or not response or identifier in found:
            raise ValueError("response rows must have unique non-empty id and response")
        found[identifier] = item
    actual = set(found)
    if actual != set(expected):
        raise ValueError("response coverage has missing or extra IDs")
    return [found[identifier] for identifier in expected]


def _load_tokenizer(model: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model), local_files_only=True)


def _vllm_answers(model: Path, prompts: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams

    engine = LLM(model=str(model), tokenizer=str(model), trust_remote_code=False)
    try:
        result = engine.generate(prompts, SamplingParams(**GENERATION), use_tqdm=False)
        answers = [item.outputs[0].text for item in result]
    finally:
        del engine
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    if len(answers) != len(prompts) or not all(isinstance(answer, str) and answer for answer in answers):
        raise ValueError("inference engine did not return one non-empty answer per prompt")
    return answers


def generate_responses(
    manifest_path: Path = MANIFEST_PATH,
    output_dir: Path = OUTPUT_ROOT,
    candidate_model: Path = CANDIDATE_MODEL,
    baseline_model: Path = BASELINE_MODEL,
    *,
    tokenizer_loader: Callable[[Path], Any] = _load_tokenizer,
    answer_generator: Callable[[Path, list[str]], list[str]] = _vllm_answers,
) -> dict[str, Path]:
    """Generate each model sequentially with the same Llama-format greedy prompts."""

    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    records = load_manifest(manifest_path)
    identifiers = [str(item["id"]) for item in records]
    frozen_tokenizer_path = Path(RUNTIME_PATHS["student_model"])
    tokenizer = tokenizer_loader(frozen_tokenizer_path)
    template = load_llama3_chat_template()
    tokenizer_sha256 = _tokenizer_hash(frozen_tokenizer_path)
    if _tokenizer_hash(Path(candidate_model)) != tokenizer_sha256 or _tokenizer_hash(Path(baseline_model)) != tokenizer_sha256:
        raise ValueError("candidate and baseline tokenizers must match the frozen Llama tokenizer")
    prompts = [format_prompt(tokenizer, item, chat_template=template) for item in records]
    rendered_hash = hashlib.sha256("\0".join(prompts).encode("utf-8")).hexdigest()
    paths: dict[str, Path] = {}
    metadata: dict[str, object] = {
        "manifest_sha256": _sha256_file(manifest_path),
        "generation_config": GENERATION,
        "generation_config_sha256": hashlib.sha256(json.dumps(GENERATION, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "prompt_sha256": rendered_hash,
        "models": {},
    }
    for label, model in (("candidate", Path(candidate_model)), ("baseline", Path(baseline_model))):
        answers = answer_generator(model, list(prompts))
        if len(answers) != TOTAL_PROMPTS:
            raise ValueError("inference engine must return exactly 768 answers")
        rows = validate_response_coverage(
            [{"id": identifier, "source": record["source"], "response": answer} for identifier, record, answer in zip(identifiers, records, answers, strict=True)],
            identifiers,
        )
        destination = output_dir / f"{label}_responses.jsonl"
        _atomic_jsonl(destination, rows)
        paths[label] = destination
        metadata["models"][label] = {
            "path": str(model),
            "model_sha256": _tree_hash(model),
            "tokenizer_sha256": tokenizer_sha256,
        }
    manifest_destination = output_dir / "response_manifest.json"
    _atomic_bytes(manifest_destination, (json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    paths["manifest"] = manifest_destination
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_MODEL)
    parser.add_argument("--baseline", type=Path, default=BASELINE_MODEL)
    args = parser.parse_args(argv)
    print(json.dumps({name: str(path) for name, path in generate_responses(args.manifest, args.output, args.candidate, args.baseline).items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
