"""Materialize immutable Magpie dataset attempts from the editable pipeline hook."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping
import stat
import sys
from typing import Any, Callable

try:  # Supports both package imports in tests and flat /task-tools execution.
    from . import pipeline
    from .async_contract import now_utc, read_status, transition_status, write_status
    from .dataset_contract import DatasetSummary, write_canonical_jsonl
    from .task_contract import ASSET_REVISIONS, RUNTIME_PATHS, load_llama3_chat_template
except ImportError:  # pragma: no cover - exercised inside the task image
    import pipeline
    from async_contract import now_utc, read_status, transition_status, write_status
    from dataset_contract import DatasetSummary, write_canonical_jsonl
    from task_contract import ASSET_REVISIONS, RUNTIME_PATHS, load_llama3_chat_template


class _UnavailableGenerator:
    def generate(self, prompts: list[str], sampling: Mapping[str, object]) -> list[str]:
        raise RuntimeError("local generator is unavailable; configure generate_new=false or install the pinned generator")


class _PinnedVllmGenerator:
    def __init__(self, engine: Any, sampling_params: Callable[..., Any]):
        self._engine = engine
        self._sampling_params = sampling_params
        self.calls: list[dict[str, object]] = []

    def generate(self, prompts: list[str], sampling: Mapping[str, object]) -> list[str]:
        if set(sampling) != {"seed", "max_tokens"}:
            raise ValueError("generator sampling must contain only seed and max_tokens")
        seed, max_tokens = sampling["seed"], sampling["max_tokens"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("generator seed must be an integer")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("generator max_tokens must be a positive integer")
        if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("generator prompts must be a list of strings")
        self.calls.append({"prompts": list(prompts), "sampling": dict(sampling)})
        parameters = self._sampling_params(seed=seed, max_tokens=max_tokens)
        outputs = self._engine.generate(list(prompts), parameters, use_tqdm=False)
        if len(outputs) != len(prompts) or any(not getattr(item, "outputs", None) for item in outputs):
            raise RuntimeError("pinned generator returned incomplete completions")
        answers = [item.outputs[0].text for item in outputs]
        if not all(isinstance(answer, str) for answer in answers):
            raise RuntimeError("pinned generator completion text must be strings")
        return answers


def load_local_generator() -> pipeline.GeneratorProtocol:
    """Load the pinned, offline local generator only for generation-enabled attempts."""

    from vllm import LLM, SamplingParams

    model = RUNTIME_PATHS["generator_model"]
    return _PinnedVllmGenerator(
        LLM(
            model=model,
            tokenizer=model,
            tensor_parallel_size=8,
            trust_remote_code=False,
        ),
        SamplingParams,
    )


EDITABLE_ROOT = Path("/app/project/data_research")
BASELINE_BUDGET_PATH = Path(RUNTIME_PATHS["baseline_budget"])


@dataclass(frozen=True)
class EditableSnapshot:
    manifest: dict[str, object]
    sha256: str
    pipeline_source_sha256: str
    pipeline_config_sha256: str
    pipeline_config_bytes: bytes


def load_pinned_tokenizer() -> Any:
    """Lazily load the task's pinned student tokenizer using its canonical template."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(RUNTIME_PATHS["student_model"], local_files_only=True)


def load_baseline_token_cap() -> int:
    payload = json.loads(BASELINE_BUDGET_PATH.read_text(encoding="utf-8"))
    value = payload.get("canonical_tokens") if isinstance(payload, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("baseline_budget.json must contain non-negative canonical_tokens")
    return value


def _source_sha256(path: Path) -> str:
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _source_manifest_payload(manifest: Mapping[str, object]) -> bytes:
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _snapshot_editable_tree(root: Path) -> EditableSnapshot:
    """Hash every ordinary editable file while rejecting all path indirection."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("editable data_research root must be an ordinary directory without symlink ancestors")
    files: list[dict[str, object]] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            child = current_path / name
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise ValueError("editable data_research tree must contain only ordinary directories and files")
        for name in names:
            child = current_path / name
            metadata = child.lstat()
            if child.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("editable data_research tree must contain only ordinary directories and files")
            files.append(
                {
                    "bytes": metadata.st_size,
                    "path": child.relative_to(root).as_posix(),
                    "sha256": _source_sha256(child),
                }
            )
    files.sort(key=lambda item: str(item["path"]).encode("utf-8"))
    manifest: dict[str, object] = {"files": files, "version": 1}
    by_path = {str(item["path"]): item for item in files}
    if "pipeline.py" not in by_path or "pipeline_config.toml" not in by_path:
        raise FileNotFoundError("editable data_research pipeline.py and pipeline_config.toml are required")
    config_path = root / "pipeline_config.toml"
    config_bytes = config_path.read_bytes()
    return EditableSnapshot(
        manifest=manifest,
        sha256=hashlib.sha256(_source_manifest_payload(manifest)).hexdigest(),
        pipeline_source_sha256=str(by_path["pipeline.py"]["sha256"]),
        pipeline_config_sha256=str(by_path["pipeline_config.toml"]["sha256"]),
        pipeline_config_bytes=config_bytes,
    )


def _load_editable_pipeline(root: Path) -> tuple[Callable[..., Iterable[dict]], dict[str, object], Path, EditableSnapshot]:
    source = root / "pipeline.py"
    snapshot = _snapshot_editable_tree(root)
    spec = importlib.util.spec_from_file_location(f"magpie_editable_pipeline_{snapshot.sha256}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load editable pipeline")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    builder = getattr(module, "build_dataset", None)
    if not callable(builder):
        raise ValueError("editable pipeline must define build_dataset")
    config = _read_toml_bytes(snapshot.pipeline_config_bytes)
    return builder, config, source, snapshot


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_attempt(
    *,
    attempt_id: str,
    hypothesis: str,
    output_dir: Path,
    config: Mapping[str, object],
    baseline_rows: Iterable[dict],
    generator: pipeline.GeneratorProtocol,
    tokenizer: Any | None = None,
    baseline_token_cap: int | None = None,
    build_dataset: Callable[..., Iterable[dict]] = pipeline.build_dataset,
    pipeline_source: Path | None = None,
    editable_root: Path | None = None,
    chat_template: str | None = None,
    editable_snapshot: EditableSnapshot | None = None,
) -> dict[str, Any]:
    """Validate and write the complete immutable artifact set for one attempt."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    existing = [item for item in destination.iterdir() if item.name != "status.json"]
    if existing:
        raise RuntimeError(f"attempt output is immutable and not empty: {destination}")
    source_root = Path(editable_root) if editable_root is not None else (Path(pipeline_source).parent if pipeline_source is not None else Path(pipeline.__file__).resolve().parent)
    snapshot = editable_snapshot if editable_snapshot is not None else _snapshot_editable_tree(source_root)
    rows = list(build_dataset(config, baseline_rows, generator))
    dataset_path = destination / "dataset.jsonl"
    summary: DatasetSummary = write_canonical_jsonl(
        dataset_path,
        rows,
        tokenizer if tokenizer is not None else load_pinned_tokenizer(),
        baseline_token_cap if baseline_token_cap is not None else load_baseline_token_cap(),
        [],
        chat_template=chat_template if chat_template is not None else load_llama3_chat_template(),
    )
    config_path = destination / "pipeline_config.toml"
    _atomic_write(config_path, snapshot.pipeline_config_bytes)
    calls = getattr(generator, "calls", [])
    provenance = {
        "attempt_id": attempt_id,
        "hypothesis": hypothesis,
        "created_at": now_utc(),
        "pipeline_source_sha256": snapshot.pipeline_source_sha256,
        "pipeline_config_sha256": snapshot.pipeline_config_sha256,
        "data_research_manifest": snapshot.manifest,
        "data_research_sha256": snapshot.sha256,
        "editable_root": str(editable_root) if editable_root is not None else None,
        "pipeline_config": dict(config),
        "asset_revisions": dict(ASSET_REVISIONS),
        "generator_calls": calls,
        "command": ["/task-tools/magpie_async_run", "submit", "--attempt-id", attempt_id, "--hypothesis", hypothesis],
        "dataset_sha256": summary.sha256,
        "records": summary.records,
        "tokens": summary.tokens,
    }
    provenance_path = destination / "provenance.json"
    _atomic_write(provenance_path, (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    manifest = {
        "records": summary.records,
        "tokens": summary.tokens,
        "dataset_sha256": summary.sha256,
        "pipeline_source_sha256": provenance["pipeline_source_sha256"],
        "pipeline_config_sha256": snapshot.pipeline_config_sha256,
        "data_research_manifest": snapshot.manifest,
        "data_research_sha256": snapshot.sha256,
    }
    manifest_path = destination / "dataset_manifest.json"
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    status_path = destination / "status.json"
    status = {"attempt_id": attempt_id, "status": "completed", "dataset_sha256": summary.sha256}
    if status_path.is_file() and read_status(status_path)["status"] == "running":
        status = transition_status(status_path, "completed", dataset_sha256=summary.sha256)
    else:
        write_status(status_path, status)
    if destination.parent.name == "attempts":
        ledger = destination.parent.parent / "experiments.jsonl"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"attempt_id": attempt_id, "hypothesis": hypothesis, "config": dict(config), "outcome": "completed", "selected": False, "dataset_sha256": summary.sha256}, sort_keys=True) + "\n")
    return status


def _read_toml(path: Path) -> dict[str, object]:
    return _read_toml_bytes(Path(path).read_bytes())


def _read_toml_bytes(payload: bytes) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("pipeline_config.toml must be UTF-8") from exc
    try:
        import tomllib
    except ImportError:
        config: dict[str, object] = {}
        in_pipeline = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                in_pipeline = line == "[pipeline]"
                continue
            if in_pipeline and "=" in line:
                key, value = (part.strip() for part in line.split("=", 1))
                config[key] = literal_eval(value.replace("true", "True").replace("false", "False"))
    else:
        document = tomllib.loads(text)
        config = document.get("pipeline")
    if not isinstance(config, dict):
        raise ValueError("pipeline_config.toml must contain a [pipeline] table")
    return config


def _load_baseline_rows() -> list[dict]:
    path = Path(os.environ.get("MAGPIE_BASELINE_JSONL", RUNTIME_PATHS["baseline_dataset"]))
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_attempt(attempt_id: str, hypothesis: str, output_dir: Path, *, editable_root: Path = EDITABLE_ROOT) -> None:
    """Run one configured attempt using only the local pinned inputs."""

    builder, config, source, snapshot = _load_editable_pipeline(Path(editable_root))
    generator: pipeline.GeneratorProtocol = load_local_generator() if config.get("generate_new") else _UnavailableGenerator()
    materialize_attempt(
        attempt_id=attempt_id,
        hypothesis=hypothesis,
        output_dir=Path(output_dir),
        config=config,
        baseline_rows=_load_baseline_rows(),
        generator=generator,
        build_dataset=builder,
        pipeline_source=source,
        editable_root=Path(editable_root),
        editable_snapshot=snapshot,
    )
