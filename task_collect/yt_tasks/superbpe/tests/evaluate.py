#!/usr/bin/env python3
"""Frozen, matched encoding-efficiency evaluator for SuperBPE artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from tokenizers import Tokenizer


EVAL_SHA256 = "81e2f413b434b5a89090f9cfeacd3ea01289bbba7439d2eabc0c6f20434b6784"
EVAL_BYTES = 944_259_457
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
REFERENCE = Path(
    "/opt/project/tokenizer_json/olmo2_p99_truncate_10G_80K_extend_200K_mw4_colon/tokenizer.json"
)
SUBMISSION = Path("/app/output/submission/tokenizer.json")
FIXED_ADDED = [
    (200000, "|||IP_ADDRESS|||", False, True),
    (200001, "<|padding|>", True, False),
    (200002, "|||EMAIL_ADDRESS|||", False, True),
    (200003, "|||PHONE_NUMBER|||", False, True),
    (200004, "<|endoftext|>", True, False),
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_artifact(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("submission/tokenizer.json must be one regular file")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("tokenizer.json exceeds the 64 MiB artifact cap")
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("model.type must be BPE")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or len(vocab) != 200_000:
        raise ValueError("BPE model vocabulary must contain exactly 200000 entries")
    if sorted(vocab.values()) != list(range(200_000)):
        raise ValueError("BPE model IDs must be exactly 0..199999")
    if model.get("dropout") is not None:
        raise ValueError("tokenizer dropout must be disabled")
    if model.get("ignore_merges") is not False:
        raise ValueError("BPE ignore_merges must be false")
    if data.get("normalizer") is not None:
        raise ValueError("normalization is frozen to None for byte-exact losslessness")
    if data.get("post_processor") is not None:
        raise ValueError("post_processor is frozen to None")
    if data.get("truncation") is not None or data.get("padding") is not None:
        raise ValueError("tokenizer-level truncation and padding must be disabled")
    decoder = data.get("decoder")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise ValueError("a ByteLevel decoder is required")
    actual = []
    for item in data.get("added_tokens", []):
        actual.append((item.get("id"), item.get("content"), item.get("special"), item.get("normalized")))
    if actual != FIXED_ADDED:
        raise ValueError("the released five added tokens, IDs, and flags must be unchanged")
    if not isinstance(data.get("pre_tokenizer"), dict):
        raise ValueError("pre_tokenizer must be defined")
    return data


def load_text(path: Path) -> tuple[str, int, str]:
    max_bytes = int(os.getenv("SUPERBPE_EVAL_MAX_BYTES", "0"))
    if max_bytes <= 0:
        if path.stat().st_size != EVAL_BYTES or sha256(path) != EVAL_SHA256:
            raise ValueError("pinned evaluation file size or SHA256 mismatch")
        raw = path.read_bytes()
        label = "full"
    else:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes)
        label = f"author_smoke_prefix_{max_bytes}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raw = raw[: exc.start]
        text = raw.decode("utf-8")
    return text, len(raw), label


def chunks_like_upstream(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    chunk_size = max(len(paragraphs) // 20, 100)
    return ["\n\n".join(paragraphs[index : index + chunk_size]) + "\n\n" for index in range(0, len(paragraphs), chunk_size)]


def main() -> None:
    validate_artifact(SUBMISSION)
    candidate = Tokenizer.from_file(str(SUBMISSION))
    reference = Tokenizer.from_file(str(REFERENCE))
    if candidate.get_vocab_size(with_added_tokens=False) != 200_000:
        raise ValueError("runtime candidate base vocabulary size mismatch")
    if candidate.get_vocab_size(with_added_tokens=True) != 200_005:
        raise ValueError("runtime candidate total vocabulary size mismatch")

    eval_path = Path(os.environ["SUPERBPE_EVAL_FILE"])
    text, byte_count, run_kind = load_text(eval_path)
    chunks = chunks_like_upstream(text)
    candidate_tokens = 0
    reference_tokens = 0
    candidate_seconds = 0.0
    reference_seconds = 0.0
    chunk_ratios = []
    for number, chunk in enumerate(chunks, 1):
        start = time.perf_counter()
        first = candidate.encode(chunk)
        candidate_seconds += time.perf_counter() - start
        if candidate.decode(first.ids, skip_special_tokens=False) != chunk:
            raise ValueError(f"candidate is not byte-exact lossless on evaluator chunk {number}")
        # Dropout is already rejected structurally. Re-encode the boundary
        # chunks as a runtime determinism probe without doubling the full
        # held-out evaluation cost.
        if number in {1, len(chunks)} and candidate.encode(chunk).ids != first.ids:
            raise ValueError(f"candidate encoding is nondeterministic on evaluator chunk {number}")
        start = time.perf_counter()
        baseline = reference.encode(chunk)
        reference_seconds += time.perf_counter() - start
        c_count, b_count = len(first.ids), len(baseline.ids)
        if c_count <= 0 or b_count <= 0:
            raise ValueError("empty token sequence on nonempty evaluation text")
        candidate_tokens += c_count
        reference_tokens += b_count
        chunk_ratios.append(b_count / c_count)
        print(f"chunk={number}/{len(chunks)} candidate_tokens={c_count} reference_tokens={b_count}", flush=True)

    candidate_bpt = byte_count / candidate_tokens
    reference_bpt = byte_count / reference_tokens
    metrics = {
        "run_kind": run_kind,
        "evaluation_protocol": "upstream paragraph chunks; token IDs discarded after each chunk",
        "eval_file_bytes": byte_count,
        "eval_file_sha256": EVAL_SHA256 if run_kind == "full" else None,
        "chunks": len(chunks),
        "candidate_token_count": candidate_tokens,
        "reference_token_count": reference_tokens,
        "bytes_per_token": candidate_bpt,
        "reference_bytes_per_token": reference_bpt,
        "bytes_per_token_ratio": candidate_bpt / reference_bpt,
        "worst_chunk_token_ratio": min(chunk_ratios),
        "candidate_encode_seconds": candidate_seconds,
        "reference_encode_seconds": reference_seconds,
        "candidate_tokenizer_sha256": sha256(SUBMISSION),
        "reference_tokenizer_sha256": sha256(REFERENCE),
    }
    if not all(math.isfinite(float(metrics[name])) and float(metrics[name]) > 0 for name in ("bytes_per_token", "reference_bytes_per_token", "bytes_per_token_ratio")):
        raise ValueError("non-finite metric")
    target = Path("/app/output/verifier-metrics.json")
    target.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
