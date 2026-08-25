"""Analyze generated tasks and solutions — summary tables and plots."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Complexity shortening ------------------------------------------------

# NOTE: "intricate" is the v2-only fourth complexity bucket added in commit
# ad9d7fe. Mirrors rl_data/generator/task_template_gen.py::TASK_COMPLEXITY.
_TASK_COMPLEXITY_ORDER = ["short", "moderate", "complex", "intricate"]
_CMD_COMPLEXITY_ORDER = ["bash-only", "bash+code", "bash+code+services"]

_CMD_COMPLEXITY_MAP = {
    "bash-only": "bash-only",
    "bash and code": "bash+code",
    "bash, code, and system services": "bash+code+services",
}

# ---- v2 axes (added in commit ad9d7fe — Verifier × Fixture × Corpus) -------
# Mirrors VERIFIER_KINDS / FIXTURE_KINDS / CORPUS_KINDS in
# rl_data.generator.task_template_gen. We pin a fixed display order so that
# legacy-only corpora and v2 corpora produce visually-comparable plots.
_VERIFIER_KIND_ORDER = [
    "exact_text",
    "metric_threshold",
    "adversarial_corpus",
    "fuzz_equivalence",
    "multi_protocol",
]
_FIXTURE_KIND_ORDER = [
    "text_only",
    "image",
    "audio",
    "video",
    "stripped_binary",
    "vendored_package",
    "multi_service_compose",
]
_CORPUS_KIND_ORDER = ["legacy", "sft_v2", "rl_v2"]
_BASE_IMAGE_ORDER = ["per_domain", "intricate"]

# Solver harness suffixes that ``rl_data.generate_solutions._summary_basename``
# may append to the per-task summary filename. Bash is the legacy / default
# and writes ``<MODEL_TAG>_summary.json`` (no suffix). Add to this tuple when
# new harnesses get wired into ``generate_solutions.py``.
_KNOWN_HARNESSES: tuple[str, ...] = ("vanillux",)


def _shorten_task_complexity(raw: str) -> str:
    m = re.match(r"(short|moderate|complex|intricate)\b", raw, re.IGNORECASE)
    return m.group(1).lower() if m else raw


def _shorten_cmd_complexity(raw: str) -> str:
    prefix = raw.split("(")[0].strip()
    return _CMD_COMPLEXITY_MAP.get(prefix, prefix)


# Default candidates reported alongside pass@1 in the banner / aggregate
# tables / quality plots. Each candidate k is included only if **at least
# one** solved task has data at that k — so a NUM_SOLUTIONS=4 smoke run will
# show only pass@4, while a uniform NUM_SOLUTIONS=8 corpus will show both
# pass@4 (intermediate) and pass@8 (ceiling). Tasks that lack a value at a
# given k contribute "-" to that cell only; pass@1 always uses every task.
_DEFAULT_PASS_K_LADDER: tuple[int, ...] = (4, 8)


def _resolve_pass_k_ladder(
    records: List[Dict[str, Any]],
    override: Optional[int] = None,
) -> List[int]:
    """Return the list of K's to report alongside pass@1.

    *override*: when set (via ``--pass-at-k``), the ladder is forced to
    ``[override]`` — useful for forcing two corpora to be compared at a
    fixed k.

    Otherwise: every k in :data:`_DEFAULT_PASS_K_LADDER` for which at least
    one solved task has data is kept. For a ragged corpus (e.g. some tasks
    at NUM_SOLUTIONS=4, others at 8) this yields ``[4, 8]`` — the pass@4
    column is dense and the pass@8 column is partial. Per-cell n-counts
    are surfaced where they differ from the bucket size.
    """
    if override is not None:
        return [int(override)]
    present: set[int] = set()
    for r in records:
        if not r.get("has_solutions"):
            continue
        present.update(r.get("pass_at_k_full", {}).keys())
    return [k for k in _DEFAULT_PASS_K_LADDER if k in present]


def _resolve_pass_k(
    records: List[Dict[str, Any]],
    override: Optional[int] = None,
) -> int:
    """Backwards-compat single-K resolver — used in spots where one number
    is genuinely needed (e.g. the vertical guide line on the pass@k curve).

    Returns the largest k that **every solved task** has data for (= min of
    per-task max-k). When all solved tasks were sampled at NUM_SOLUTIONS=K,
    this is just K.
    """
    if override is not None:
        return int(override)
    ks_max = [
        r.get("pass_k_max_avail", 0)
        for r in records
        if r.get("has_solutions") and r.get("pass_k_max_avail", 0) > 0
    ]
    if not ks_max:
        return 0
    return min(ks_max)


def _split_harness(slug: str) -> tuple[str, str]:
    """Split a slug discovered from a ``*_summary.json`` filename into
    ``(model_slug, harness)``.

    The naming convention (see ``rl_data.generate_solutions._summary_basename``)
    is:
    * ``<MODEL_TAG>_summary.json``               → harness = ``"bash"``  (legacy)
    * ``<MODEL_TAG>_<HARNESS>_summary.json``     → harness = ``<HARNESS>``

    We only strip *known* harness suffixes so model slugs that happen to end
    in things like ``_preview`` keep working unchanged.
    """
    for h in _KNOWN_HARNESSES:
        suffix = f"_{h}"
        if slug.endswith(suffix):
            return slug[: -len(suffix)], h
    return slug, "bash"


def _is_task_dir(p: Path) -> bool:
    """Predicate that recognizes both native `task_*` dirs and adapter-
    produced dirs (``otrl_task_*``, ``otb_*``, ...). Mirrors the one used by
    ``rl_data.generate_solutions`` and ``rl_data.comparison.taxonomy_classifier``."""
    return p.is_dir() and (p.name.startswith("task_") or (p / "task.json").exists())


def discover_models(tasks_dir: Path) -> List[tuple[str, str]]:
    """Return sorted list of ``(model_slug, harness)`` pairs found across all
    task solution dirs.

    Vanillux (and any other future non-bash harness) writes summaries as
    ``<MODEL>_<HARNESS>_summary.json`` so that bash and vanillux runs against
    the same model can coexist in one task dir. We surface each (model,
    harness) pair as its own analysis target so the per-axis plots stay
    apples-to-apples within a harness.
    """
    pairs: set[tuple[str, str]] = set()
    for task_path in tasks_dir.iterdir():
        if not _is_task_dir(task_path):
            continue
        solutions_dir = task_path / "solutions"
        if not solutions_dir.exists():
            continue
        for f in solutions_dir.glob("*_summary.json"):
            if f.name == "summary.json":
                continue
            raw_slug = f.name.removesuffix("_summary.json")
            pairs.add(_split_harness(raw_slug))
    return sorted(pairs)


_TOK_PER_WORD = 1.3  # rough whitespace-word → BPE-token ratio

# (input_$/1M_tok, output_$/1M_tok) — text only
# slug → (price_in, price_out).  Slug is model_id with "/" replaced by "_".
# Source: https://ai.google.dev/gemini-api/docs/gemini-3
_PRICING: Dict[str, tuple] = {
    "gemini_gemini-3.1-pro-preview":        (2.00, 12.00),
    "gemini_gemini-3-pro-preview":          (2.00, 12.00),
    "gemini_gemini-3-flash-preview":        (0.50,  3.00),
    "gemini_gemini-3.1-flash-lite-preview": (0.25,  1.50),
    "gemini_gemini-2.5-pro":                (1.25, 10.00),
    "gemini_gemini-2.5-flash":              (0.15,  0.60),
    "gemini_gemini-2.0-flash":              (0.10,  0.40),
}

# Task generation always uses 3.1 Pro
_TASK_GEN_MODEL = "gemini_gemini-3.1-pro-preview"
_TASK_GEN_AVG_INPUT_WORDS = 800
_TASK_GEN_AVG_OUTPUT_WORDS = 1200


def _estimate_cost(input_tokens: int, output_tokens: int,
                   model_slug: str) -> float:
    """Return estimated USD cost given token counts and model slug."""
    pi, po = _PRICING.get(model_slug, (1.0, 5.0))
    return (input_tokens * pi + output_tokens * po) / 1e6


def _msg_words(msg: Dict[str, Any]) -> int:
    """Total word count in a single chat message (content + tool-call args)."""
    content = msg.get("content")
    wc = len(content.split()) if content else 0
    for tc in (msg.get("tool_calls") or []):
        args = tc.get("function", {}).get("arguments", "")
        if args:
            wc += len(args.split())
    return wc


def _count_words_in_messages(
    messages: List[Dict[str, Any]],
) -> tuple:
    """Return (input_words, output_words) across all messages."""
    inp, out = 0, 0
    for msg in messages:
        w = _msg_words(msg)
        if msg.get("role") == "assistant":
            out += w
        else:
            inp += w
    return inp, out


def _peak_context_words(
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Decompose a run's messages into turn-level word counts.

    Returns a dict with:
      - sys_user_w:     words in leading system + first user message(s)
      - turn_pair_words: per-turn words of (assistant_i + tool_i) for i<T
                         (the new content added to the history after each
                         assistant turn; tool/user content that follows is
                         attributed to the preceding assistant's turn-pair)
      - last_asst_w:    words in the final assistant message
      - prompt_sum_w:   total "input" words summed over all T per-turn prompts
                        (each prompt = system+user + all prior turn-pairs)
      - completion_sum_w: total assistant words across all T turns
      - num_turns:      T (= number of assistant messages)
    Returns None if the run has no assistant message.
    """
    i = 0
    sys_user_w = 0
    while i < len(messages) and messages[i].get("role") in ("system", "user"):
        sys_user_w += _msg_words(messages[i])
        i += 1

    turn_pair_words: List[int] = []
    current_asst_w = 0
    current_extra_w = 0
    saw_asst = False
    asst_word_list: List[int] = []

    for m in messages[i:]:
        role = m.get("role")
        w = _msg_words(m)
        if role == "assistant":
            if saw_asst:
                turn_pair_words.append(current_asst_w + current_extra_w)
                current_extra_w = 0
            current_asst_w = w
            asst_word_list.append(w)
            saw_asst = True
        else:
            if saw_asst:
                current_extra_w += w
    if saw_asst:
        turn_pair_words.append(current_asst_w + current_extra_w)

    T = len(asst_word_list)
    if T == 0:
        return None

    prompt_sum_w = T * sys_user_w
    for j in range(T - 1):
        prompt_sum_w += (T - 1 - j) * turn_pair_words[j]
    completion_sum_w = sum(asst_word_list)

    return {
        "sys_user_w": sys_user_w,
        "turn_pair_words": turn_pair_words,
        "last_asst_w": asst_word_list[-1],
        "prompt_sum_w": prompt_sum_w,
        "completion_sum_w": completion_sum_w,
        "num_turns": T,
    }


# ── Optional infra-error exclusion ─────────────────────────────────────────
# When COMPARISON_EXCLUDE_INFRA=1, individual rollout runs whose verifier never
# produced a fair pass/fail verdict (harness/infra failures, NOT genuine model
# failures) are dropped before pass@k is recomputed. A task left with zero valid
# runs is treated as "no solution" (dropped from the comparison entirely) rather
# than counted as a 0. The signatures below are deliberately HIGH-PRECISION so
# the filter is safe to apply across every dataset: a genuine test failure always
# carries a pytest "N passed/failed" verdict and is never flagged.
_EXCLUDE_INFRA = os.environ.get("COMPARISON_EXCLUDE_INFRA", "").strip().lower() in (
    "1", "true", "yes", "on",
)

# Substrings that unambiguously indicate a harness/verifier-side failure (the
# verifier could not run), independent of anything the agent did.
_INFRA_MARKERS = (
    "Fatal Python error",            # interpreter could not boot at all
    "INTERNALERROR",                 # pytest crashed internally
    "cannot contain null bytes",     # corrupted site/.pth bootstrap
)


def _run_is_infra(result: Dict[str, Any]) -> bool:
    """True if a single rollout run failed for harness/infra reasons rather
    than a genuine (verifier-confirmed) model failure.

    Conservative by design: success always counts as valid, and any run that
    carries a real pytest verdict (``N passed`` / ``N failed`` / ``N error``)
    counts as valid even if it also prints a traceback.
    """
    if result.get("success"):
        return False
    out = (result.get("output") or "").strip()
    if not out:
        # No verifier signal at all — the verifier produced nothing.
        return True
    # PRECEDENCE: a real pytest verdict ("N passed" / "N failed") means the
    # verifier actually ran and judged the agent's work — that is a GENUINE
    # model result, even if the body also contains a FileNotFoundError (the
    # agent's own missing output) or a traceback inside an assertion message.
    # Only runs that never reached a verdict can be infra.
    if re.search(r"\b\d+ (passed|failed)\b", out):
        return False
    # No verdict reached → the verifier never produced a result. Flag the
    # harness/bootstrap failure signatures we recognise.
    if "No such file or directory" in out and (
        "pytest_final_state" in out or "test_final_state" in out or "agent_env" in out
    ):
        return True
    if any(m in out for m in _INFRA_MARKERS):
        return True
    # A bare interpreter/verifier-bootstrap traceback with no verdict means the
    # test session never ran (e.g. the agent left the conda/toolchain corrupted
    # so pytest itself can't import).
    if "Traceback (most recent call last)" in out:
        return True
    return False


def _apply_infra_filter(sol: Dict[str, Any]) -> None:
    """In-place: drop infra runs from *sol* and recompute num_runs/num_success/
    pass_at_k over the remaining valid runs (unbiased estimator, matching
    rl_data.generator.vanillux_solver)."""
    results = sol.get("results", [])
    valid = [r for r in results if not _run_is_infra(r)]
    if len(valid) == len(results):
        return  # nothing infra — leave the summary untouched
    n = len(valid)
    c = sum(1 for r in valid if r.get("success"))
    sol["results"] = valid
    sol["num_runs"] = n
    sol["num_success"] = c
    pak: Dict[int, float] = {}
    for k in range(1, n + 1):
        pak[k] = 0.0 if c == 0 else float(1.0 - (comb(n - c, k) / comb(n, k)))
    sol["pass_at_k"] = pak


def _load_summary(summary_path: Path, record: Dict[str, Any]) -> None:
    """Populate *record* with metrics from a model summary file.

    If the summary contains actual ``usage`` data (prompt_tokens,
    completion_tokens) captured from the API, those are used directly.
    Otherwise we fall back to word-count estimation.
    """
    with open(summary_path) as f:
        sol = json.load(f)
    if _EXCLUDE_INFRA:
        _apply_infra_filter(sol)
    record["num_runs"] = sol.get("num_runs", 0)
    record["num_success"] = sol.get("num_success", 0)
    raw_pak = sol.get("pass_at_k", {})
    # Coerce keys to ``int`` once so downstream code doesn't have to fight
    # the JSON-string-vs-int duality.  ``run_n_solutions`` writes ``int``,
    # ``run_n_solutions_vanillux`` also writes ``int``, but ``json.dumps`` /
    # ``json.loads`` of a dict-keyed-by-int converts the keys to strings on
    # disk, so what we read back is always strings. Be defensive.
    pass_at_k_full: Dict[int, float] = {}
    for k, v in raw_pak.items():
        try:
            pass_at_k_full[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    record["pass_at_k_full"] = pass_at_k_full
    record["pass_k_max_avail"] = max(pass_at_k_full) if pass_at_k_full else 0
    record["pass@1"] = pass_at_k_full.get(1)
    # ``harness`` was added to summaries by vanillux_solver in commit ad9d7fe.
    # Bash summaries don't carry the field, so we infer "bash" by default and
    # fall back to filename-based detection if the JSON omits it.
    if "harness" in sol:
        record["harness"] = sol["harness"]
    else:
        _, h_from_name = _split_harness(
            summary_path.name.removesuffix("_summary.json")
        )
        record["harness"] = h_from_name

    turns_per_run = []
    input_words_per_run = []
    output_words_per_run = []
    peak_info_per_run: List[Optional[Dict[str, Any]]] = []
    for r in sol.get("results", []):
        msgs = r.get("messages", [])
        n_turns = sum(1 for m in msgs if m.get("role") == "tool")
        turns_per_run.append(n_turns)
        iw, ow = _count_words_in_messages(msgs)
        input_words_per_run.append(iw)
        output_words_per_run.append(ow)
        peak_info_per_run.append(_peak_context_words(msgs))

    n = len(turns_per_run)
    total_in_w = sum(input_words_per_run)
    total_out_w = sum(output_words_per_run)

    record["avg_turns"] = sum(turns_per_run) / n if n else 0
    record["total_words"] = total_in_w + total_out_w
    record["avg_words"] = record["total_words"] / n if n else 0

    # Prefer actual API usage when available; fall back to word-count estimate.
    top_usage = sol.get("usage")
    has_actual = (
        top_usage
        and (top_usage.get("prompt_tokens", 0) or top_usage.get("completion_tokens", 0))
    )

    if has_actual:
        record["total_input_tokens"] = top_usage.get("prompt_tokens", 0)
        record["total_output_tokens"] = top_usage.get("completion_tokens", 0)
        record["reasoning_tokens"] = top_usage.get("reasoning_tokens", 0)
        record["tokens_source"] = "api"
    else:
        # Try summing per-result usage (newer format may have per-result but no top-level)
        per_result_in = 0
        per_result_out = 0
        found_any = False
        for r in sol.get("results", []):
            ru = r.get("usage")
            if ru and (ru.get("prompt_tokens", 0) or ru.get("completion_tokens", 0)):
                per_result_in += ru.get("prompt_tokens", 0)
                per_result_out += ru.get("completion_tokens", 0)
                found_any = True
        if found_any:
            record["total_input_tokens"] = per_result_in
            record["total_output_tokens"] = per_result_out
            record["reasoning_tokens"] = sum(
                (r.get("usage") or {}).get("reasoning_tokens", 0)
                for r in sol.get("results", [])
            )
            record["tokens_source"] = "api"
        else:
            record["total_input_tokens"] = int(total_in_w * _TOK_PER_WORD)
            record["total_output_tokens"] = int(total_out_w * _TOK_PER_WORD)
            record["reasoning_tokens"] = 0
            record["tokens_source"] = "estimated"

    record["total_tokens_est"] = (
        record["total_input_tokens"] + record["total_output_tokens"]
    )
    record["avg_tokens_est"] = record["total_tokens_est"] // n if n else 0

    # ── Per-run peak-context estimates (final turn input / output, initial
    # input), averaged across runs for this task.  Stored tokens-per-word is
    # calibrated per-run to the recorded API usage when available so that
    # summing per-turn inputs reproduces the recorded per-run prompt_tokens.
    initial_in_sum = 0.0
    peak_in_sum = 0.0
    final_out_sum = 0.0
    n_peak = 0
    for r, info in zip(sol.get("results", []), peak_info_per_run):
        if not info:
            continue
        tpw: Optional[float] = None
        ru = r.get("usage") or {}
        pt = ru.get("prompt_tokens", 0) or 0
        ct = ru.get("completion_tokens", 0) or 0
        denom_w = info["prompt_sum_w"] + info["completion_sum_w"]
        if (pt + ct) > 0 and denom_w > 0:
            tpw = (pt + ct) / denom_w
        elif denom_w > 0:
            tpw = _TOK_PER_WORD
        if tpw is None:
            continue
        T = info["num_turns"]
        sys_user_w = info["sys_user_w"]
        pre_last_w = sys_user_w + sum(info["turn_pair_words"][: T - 1])
        initial_in_sum += sys_user_w * tpw
        peak_in_sum += pre_last_w * tpw
        final_out_sum += info["last_asst_w"] * tpw
        n_peak += 1

    if n_peak:
        record["avg_initial_input_tokens"] = initial_in_sum / n_peak
        record["avg_peak_input_tokens"] = peak_in_sum / n_peak
        record["avg_final_output_tokens"] = final_out_sum / n_peak
    else:
        record["avg_initial_input_tokens"] = 0.0
        record["avg_peak_input_tokens"] = 0.0
        record["avg_final_output_tokens"] = 0.0

    # With the infra filter on, a task whose every run was an infra failure has
    # no valid runs left (n == 0); drop it from the comparison rather than
    # scoring it as a 0 (the model was never fairly evaluated).
    record["has_solutions"] = n > 0


def _summary_basename(model_slug: str, harness: str) -> str:
    """Mirror of ``rl_data.generate_solutions._summary_basename`` operating
    on slugs (model_id with ``/`` already replaced by ``_``)."""
    if harness == "bash":
        return f"{model_slug}_summary.json"
    return f"{model_slug}_{harness}_summary.json"


def load_tasks(
    tasks_dir: Path,
    model_slug: Optional[str] = None,
    harness: str = "bash",
) -> List[Dict[str, Any]]:
    """Scan *tasks_dir* for task directories and load metadata + solution summaries.

    If *model_slug* is given (e.g. ``"gemini_gemini-3-flash-preview"``), only
    that model + harness pair's summary is loaded (file named
    ``<slug>_summary.json`` for bash, ``<slug>_<HARNESS>_summary.json`` for
    others — see ``rl_data.generate_solutions._summary_basename``).
    Otherwise the first ``*_summary.json`` found is used.
    """
    records = []
    for task_path in sorted(tasks_dir.iterdir()):
        if not _is_task_dir(task_path):
            continue
        task_json = task_path / "task.json"
        if not task_json.exists():
            continue

        with open(task_json) as f:
            task_data = json.load(f)

        # Adapter-produced tasks (ET, OT-Agent-v1-RL, ...) leave the native
        # taxonomy fields as "unknown" and get LLM-classified values written
        # under `classified_*` by rl_data.comparison.taxonomy_classifier.
        # Prefer those when present, falling back to native fields so
        # skill-tax (which has native taxonomy) keeps working.
        def _pref(*keys: str, default: str = "unknown") -> str:
            for k in keys:
                v = task_data.get(k)
                if v not in (None, "", "unknown"):
                    return v
            return default

        raw_tc = _pref("classified_task_complexity", "task_complexity", "complexity")
        raw_cc = _pref("classified_command_complexity", "command_complexity")

        # v2 axes (commit ad9d7fe). Defaults reproduce the legacy bucket so
        # pre-v2 task.json files (which simply omit these keys) get the same
        # categorical treatment as a corpus_kind="legacy" v2 task.
        verifier_kind = task_data.get("verifier_kind") or "exact_text"
        fixture_kind = task_data.get("fixture_kind") or "text_only"
        corpus_kind = task_data.get("corpus_kind") or "legacy"
        # task_template_gen sets base_image="intricate" only for v2 tasks
        # (any non-legacy verifier/fixture or intricate complexity); pre-v2
        # tasks leave the field unset, which we surface as "per_domain".
        base_image = task_data.get("base_image") or "per_domain"

        record: Dict[str, Any] = {
            "name": task_data.get("name", task_path.name),
            "domain": _pref("classified_domain", "domain", "category"),
            "skill_type": _pref("classified_skill_type", "skill_type"),
            "primitive_skills": (task_data.get("classified_primitive_skills")
                                 or task_data.get("primitive_skills", [])),
            "task_complexity": _shorten_task_complexity(raw_tc),
            "command_complexity": _shorten_cmd_complexity(raw_cc),
            "scenario": _pref("classified_scenario", "scenario"),
            # v2-axis fields — always populated, default to legacy values.
            "verifier_kind": verifier_kind,
            "fixture_kind": fixture_kind,
            "corpus_kind": corpus_kind,
            "base_image": base_image,
            "dir": str(task_path),
        }

        _NO_SOLUTION = dict(
            num_runs=0, num_success=0, avg_turns=0,
            total_words=0, avg_words=0,
            total_input_tokens=0, total_output_tokens=0,
            reasoning_tokens=0,
            total_tokens_est=0, avg_tokens_est=0,
            avg_initial_input_tokens=0.0,
            avg_peak_input_tokens=0.0,
            avg_final_output_tokens=0.0,
            tokens_source="none",
            has_solutions=False,
            harness="none",
            pass_at_k_full={},
            pass_k_max_avail=0,
            **{"pass@1": None},
        )

        solutions_dir = task_path / "solutions"
        if model_slug:
            summary_file = solutions_dir / _summary_basename(model_slug, harness)
            if summary_file.exists():
                _load_summary(summary_file, record)
                record.setdefault("harness", harness)
            else:
                record.update(_NO_SOLUTION)
        else:
            summary_files = (
                list(solutions_dir.glob("*_summary.json"))
                if solutions_dir.exists()
                else []
            )
            summary_files = [f for f in summary_files if f.name != "summary.json"]
            if summary_files:
                _load_summary(summary_files[0], record)
            else:
                record.update(_NO_SOLUTION)

        records.append(record)
    return records


def _fmt_count(n: int) -> str:
    """Human-friendly large number: 1234 → '1,234', 1234567 → '1.23M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _pass_at_k(record: Dict[str, Any], k: int) -> Optional[float]:
    """Look up pass@k for a record (returns None if the task has no value at
    this specific k — e.g. it was sampled at NUM_SOLUTIONS=4 and we asked
    for k=8)."""
    if not record.get("has_solutions"):
        return None
    return record.get("pass_at_k_full", {}).get(k)


def print_summary(
    records: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    model_slug: Optional[str] = None,
    max_rows: int = 50,
    harness: str = "bash",
    pass_k_ladder: Optional[List[int]] = None,
) -> None:
    """Print model stats banner, optionally per-task rows, and aggregate table.

    *pass_k_ladder* is the list of K's to report alongside pass@1. Resolved
    by ``_resolve_pass_k_ladder`` — defaults to every K in
    :data:`_DEFAULT_PASS_K_LADDER` for which at least one solved task has
    data. Tasks without a value at a given K are excluded from that
    specific cell only (per-cell n-counts are shown when they differ from
    the bucket size).
    """
    if pass_k_ladder is None:
        pass_k_ladder = []
    solved = [r for r in records if r["has_solutions"]]

    # ── Model-level stats banner ──────────────────────────────────────
    total_runs = sum(r["num_runs"] for r in records)
    total_words = sum(r["total_words"] for r in records)
    total_in_tok = sum(r["total_input_tokens"] for r in records)
    total_out_tok = sum(r["total_output_tokens"] for r in records)
    total_tokens = total_in_tok + total_out_tok
    avg_words = int(total_words / total_runs) if total_runs else 0
    avg_tokens = int(total_tokens / total_runs) if total_runs else 0

    title = model_name or "all models"
    if harness and harness != "bash":
        title = f"{title}  [harness: {harness}]"
    print(f"\n{'═'*80}")
    print(f"  {title}")
    print(f"{'═'*80}")
    print(
        f"  Tasks: {len(records)}  "
        f"│  With solutions: {len(solved)}  "
        f"│  Total runs: {total_runs}"
    )

    # Surface the per-corpus k mix so it's obvious when a corpus is ragged
    # (e.g. some tasks sampled at NUM_SOLUTIONS=4, others at 8).
    if solved:
        ks_max = Counter(r["pass_k_max_avail"] for r in solved)
        ladder_str = ", ".join(f"k={k}" for k in pass_k_ladder) or "(none)"
        if len(ks_max) > 1:
            mix = ", ".join(
                f"k≤{k}: {n}" for k, n in sorted(ks_max.items())
            )
            print(
                f"  Pass@k ladder: {ladder_str}  "
                f"(per-task max-k mix: {mix})"
            )
        else:
            (only_k,) = list(ks_max)
            print(f"  Pass@k ladder: {ladder_str}  "
                  f"(every solved task has k=1..{only_k})")

    if solved:
        avg_p1 = (
            sum(r["pass@1"] for r in solved if r["pass@1"] is not None)
            / len(solved)
        )
        avg_turns = sum(r["avg_turns"] for r in solved) / len(solved)
        bits = [f"Mean p@1: {avg_p1:.2f}"]
        for k in pass_k_ladder:
            pk_vals = [v for v in (_pass_at_k(r, k) for r in solved) if v is not None]
            if not pk_vals:
                bits.append(f"Mean p@{k}: -")
                continue
            avg_pk = sum(pk_vals) / len(pk_vals)
            n_lbl = (
                f" (n={len(pk_vals)}/{len(solved)})"
                if len(pk_vals) != len(solved) else ""
            )
            bits.append(f"Mean p@{k}: {avg_pk:.2f}{n_lbl}")
        bits.append(f"Avg turns: {avg_turns:.1f}")
        print("  " + "  │  ".join(bits))
    print(
        f"  Total words: {_fmt_count(total_words)}  "
        f"│  Avg words/run: {_fmt_count(avg_words)}  "
        f"│  Avg tokens/run: {_fmt_count(avg_tokens)}"
    )
    # Determine token source label
    sources = {r.get("tokens_source", "none") for r in records if r["has_solutions"]}
    if sources == {"api"}:
        tok_label = "actual"
    elif "api" in sources:
        tok_label = "actual+est"
    else:
        tok_label = "est"

    total_reasoning = sum(r.get("reasoning_tokens", 0) for r in records)
    reasoning_info = ""
    if total_reasoning:
        reasoning_info = f"  (incl. {_fmt_count(total_reasoning)} reasoning)"

    print(
        f"  Input tokens: {_fmt_count(total_in_tok)}  "
        f"│  Output tokens: {_fmt_count(total_out_tok)}  "
        f"│  Total tokens: {_fmt_count(total_tokens)}  ({tok_label})"
    )
    if reasoning_info:
        print(f"  {reasoning_info}")

    # Cost estimation
    slug = model_slug
    if slug and slug in _PRICING:
        pi, po = _PRICING[slug]
        sol_cost = _estimate_cost(total_in_tok, total_out_tok, slug)

        # Task generation cost (always 3.1 Pro)
        n_tasks = len(records)
        tg_in = int(n_tasks * _TASK_GEN_AVG_INPUT_WORDS * _TOK_PER_WORD)
        tg_out = int(n_tasks * _TASK_GEN_AVG_OUTPUT_WORDS * _TOK_PER_WORD)
        tg_cost = _estimate_cost(tg_in, tg_out, _TASK_GEN_MODEL)

        cost_per_run = sol_cost / total_runs if total_runs else 0
        print(
            f"  Solution cost (est): ${sol_cost:,.2f}  "
            f"(${pi:.2f}/${po:.2f} per 1M tok)  "
            f"│  $/run: ${cost_per_run:,.4f}"
        )
        print(
            f"  Task gen cost (est): ${tg_cost:,.2f}  "
            f"(3.1-pro, {n_tasks} tasks)  "
            f"│  Total pipeline: ${tg_cost + sol_cost:,.2f}"
        )
    elif slug:
        print(f"  Cost: pricing not available for {slug}")

    print(f"{'─'*80}")

    # ── Per-task rows (skip if too many) ──────────────────────────────
    show_rows = max_rows == 0 or len(records) <= max_rows
    if show_rows:
        pk_cols = " ".join(f"{f'p@{k}':>6}" for k in pass_k_ladder)
        header = (
            f"{'Task':<30} {'Domain':<24} {'Skill Type':<20} "
            f"{'Task Cplx':<12} {'Cmd Cplx':<20} "
            f"{'Runs':>5} {'Pass':>5} "
            f"{'p@1':>6} {pk_cols} {'Turns':>6}".rstrip()
        )
        print(header)
        print("-" * len(header))
        for r in records:
            p1 = f"{r['pass@1']:.2f}" if r["pass@1"] is not None else "-"
            pk_cells = []
            for k in pass_k_ladder:
                v = _pass_at_k(r, k)
                pk_cells.append(f"{v:>6.2f}" if v is not None else f"{'-':>6}")
            pk_str = " ".join(pk_cells)
            turns = (
                f"{r['avg_turns']:>6.1f}" if r["has_solutions"] else f"{'-':>6}"
            )
            print(
                f"{r['name']:<30} {r['domain']:<24} {r['skill_type']:<20} "
                f"{r['task_complexity']:<12} {r['command_complexity']:<20} "
                f"{r['num_runs']:>5} {r['num_success']:>5} "
                f"{p1:>6} {pk_str} {turns}".rstrip()
            )
        print()
    else:
        print(
            f"  ({len(records)} tasks — per-task rows hidden; "
            f"use --max-rows 0 to show all)\n"
        )

    # ── Aggregate breakdown ───────────────────────────────────────────
    if not solved:
        return
    _print_aggregate(solved, "domain", "Domain", model_slug=slug,
                     pass_k_ladder=pass_k_ladder)
    _print_aggregate(solved, "task_complexity", "Task Complexity",
                     key_order=_TASK_COMPLEXITY_ORDER, model_slug=slug,
                     pass_k_ladder=pass_k_ladder)
    _print_aggregate(solved, "command_complexity", "Cmd Complexity",
                     key_order=_CMD_COMPLEXITY_ORDER, model_slug=slug,
                     pass_k_ladder=pass_k_ladder)

    # ── v2-axis breakdowns (only print when the axis isn't degenerate) ─
    # Suppresses pure noise on legacy-only corpora where every task has the
    # same default value, while still surfacing the breakdown the moment a
    # corpus mixes legacy + v2 tasks.
    def _has_variation(field: str) -> bool:
        return len({r[field] for r in solved}) > 1

    if _has_variation("verifier_kind"):
        _print_aggregate(solved, "verifier_kind", "Verifier Kind",
                         key_order=_VERIFIER_KIND_ORDER, model_slug=slug,
                         pass_k_ladder=pass_k_ladder)
    if _has_variation("fixture_kind"):
        _print_aggregate(solved, "fixture_kind", "Fixture Kind",
                         key_order=_FIXTURE_KIND_ORDER, model_slug=slug,
                         pass_k_ladder=pass_k_ladder)
    if _has_variation("corpus_kind"):
        _print_aggregate(solved, "corpus_kind", "Corpus Kind",
                         key_order=_CORPUS_KIND_ORDER, model_slug=slug,
                         pass_k_ladder=pass_k_ladder)
    if _has_variation("base_image"):
        _print_aggregate(solved, "base_image", "Base Image",
                         key_order=_BASE_IMAGE_ORDER, model_slug=slug,
                         pass_k_ladder=pass_k_ladder)


def _print_aggregate(
    solved: List[Dict[str, Any]],
    field: str,
    label: str,
    key_order: Optional[List[str]] = None,
    model_slug: Optional[str] = None,
    pass_k_ladder: Optional[List[int]] = None,
) -> None:
    """Print a small aggregate table grouped by *field*.

    Each K in *pass_k_ladder* gets its own ``p@K`` column. Tasks in a
    bucket that don't have a value at a given K are excluded from that
    cell's average only — pass@1 (and bucket size ``n``) always uses every
    task, so a ragged corpus is obvious from the difference.
    """
    if pass_k_ladder is None:
        pass_k_ladder = []
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in solved:
        buckets[r[field]].append(r)

    keys = key_order if key_order else sorted(buckets.keys())
    has_cost = model_slug and model_slug in _PRICING

    pk_hdr = " ".join(f"{f'p@{k}':>6}" for k in pass_k_ladder)
    pk_dash_blocks = " ".join(f"{'-':>6}" for _ in pass_k_ladder)
    hdr = (
        f"  {'':2}{label:<24} {'n':>4} {'p@1':>6} {pk_hdr} "
        f"{'Turns':>6} {'Tokens':>10}"
    ).rstrip()
    if has_cost:
        hdr += f" {'Cost':>10}"
    print(hdr)
    print(f"  {'':2}{'-'*(len(hdr)-4)}")
    for k_label in keys:
        recs = buckets.get(k_label, [])
        if not recs:
            empty_pk = pk_dash_blocks if pk_dash_blocks else ""
            line = (f"  {'':2}{k_label:<24} {'0':>4} {'-':>6} {empty_pk} "
                    f"{'-':>6} {'-':>10}").rstrip()
            if has_cost:
                line += f" {'-':>10}"
            print(line)
            continue
        n = len(recs)
        mp1 = sum(r["pass@1"] for r in recs if r["pass@1"] is not None) / n
        pk_cells: List[str] = []
        for k in pass_k_ladder:
            pk_vals = [v for v in (_pass_at_k(r, k) for r in recs) if v is not None]
            if pk_vals:
                pk_cells.append(f"{sum(pk_vals) / len(pk_vals):>6.2f}")
            else:
                pk_cells.append(f"{'-':>6}")
        pk_str = " ".join(pk_cells)
        mt = sum(r["avg_turns"] for r in recs) / n
        ti = sum(r["total_input_tokens"] for r in recs)
        to = sum(r["total_output_tokens"] for r in recs)
        tt = ti + to
        line = (
            f"  {'':2}{k_label:<24} {n:>4} {mp1:>6.2f} {pk_str} "
            f"{mt:>6.1f} {_fmt_count(tt):>10}"
        ).rstrip()
        if has_cost:
            c = _estimate_cost(ti, to, model_slug)
            line += f" {'${:,.2f}'.format(c):>10}"
        print(line)
    print()


def plot_distributions(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate pie charts for all metadata axes (legacy + v2)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legacy axes — always plotted for backwards-compatibility.
    axes = [
        ("domain", "Domain Distribution", "dist_domain.png", None),
        ("skill_type", "Skill Type Distribution", "dist_skill_type.png", None),
        ("task_complexity", "Task Complexity Distribution",
         "dist_task_complexity.png", _TASK_COMPLEXITY_ORDER),
        (
            "command_complexity",
            "Command Complexity Distribution",
            "dist_command_complexity.png",
            _CMD_COMPLEXITY_ORDER,
        ),
        ("scenario", "Scenario Distribution", "dist_scenario.png", None),
    ]

    # v2 axes (commit ad9d7fe). Always plotted — for a legacy-only corpus
    # the verifier/fixture/corpus pies will simply collapse to a single
    # "exact_text" / "text_only" / "legacy" / "per_domain" wedge, which is
    # itself useful as a sanity check.
    axes.extend([
        ("verifier_kind", "Verifier Kind Distribution",
         "dist_verifier_kind.png", _VERIFIER_KIND_ORDER),
        ("fixture_kind", "Fixture Kind Distribution",
         "dist_fixture_kind.png", _FIXTURE_KIND_ORDER),
        ("corpus_kind", "Corpus Kind Distribution",
         "dist_corpus_kind.png", _CORPUS_KIND_ORDER),
        ("base_image", "Base Image Distribution",
         "dist_base_image.png", _BASE_IMAGE_ORDER),
    ])

    for field, title, fname, order in axes:
        counts = Counter(r[field] for r in records)
        if order is not None:
            ordered_keys = [k for k in order if counts.get(k)]
            extra_keys = sorted(k for k in counts if k not in order)
            labels = ordered_keys + extra_keys
        else:
            labels = list(counts.keys())
        sizes = [counts[k] for k in labels]
        if not sizes:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, _texts, _autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.0f%%",
            startangle=90,
            pctdistance=0.85,
            textprops={"fontsize": 9},
        )
        ax.legend(
            wedges,
            [f"{lb} ({ct})" for lb, ct in zip(labels, sizes)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / fname}")


def _bar_chart(
    records: List[Dict[str, Any]],
    field: str,
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
    out_dir: Path,
    color: str = "steelblue",
    expected_keys: Optional[List[str]] = None,
) -> None:
    """Helper: grouped bar chart of *metric* averaged by *field*.

    If *expected_keys* is given, all listed categories are shown (in that
    order) even when no data exists for some of them.
    """
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        val = r.get(metric)
        if val is not None:
            buckets[r[field]].append(val)
    if not buckets and not expected_keys:
        return

    if expected_keys:
        keys = expected_keys
    else:
        keys = sorted(buckets.keys())

    means = [
        (sum(buckets[k]) / len(buckets[k])) if buckets.get(k) else 0
        for k in keys
    ]
    counts = [len(buckets.get(k, [])) for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    bars = ax.bar(range(len(keys)), means, color=color)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if metric.startswith("pass"):
        ax.set_ylim(0, 1.05)
    for bar, val, n in zip(bars, means, counts):
        if n == 0:
            label = "n=0"
        elif metric.startswith("pass"):
            label = f"{val:.2f}\n(n={n})"
        else:
            label = f"{val:.1f}\n(n={n})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / fname}")


def plot_quality(
    records: List[Dict[str, Any]],
    out_dir: Path,
    model_name: Optional[str] = None,
    model_slug: Optional[str] = None,
    harness: str = "bash",
    pass_k_ladder: Optional[List[int]] = None,
) -> None:
    """Generate quality analysis plots (bar charts + pass@k curve).

    For each K in *pass_k_ladder* (e.g. ``[4, 8]`` on a uniform-NUM_SOLUTIONS=8
    corpus) we emit one set of bar charts (``quality_pass{K}_by_domain.png``,
    ...). pass@1 charts are always emitted. A separate
    ``quality_num_success_distribution.png`` shows the per-num_runs
    histogram of ``num_success`` (i.e. how many tasks got 0/N right, 1/N,
    ..., N/N), which makes a corpus's "easy / medium / hard" split visible
    at a glance.
    """
    if pass_k_ladder is None:
        pass_k_ladder = []
    out_dir.mkdir(parents=True, exist_ok=True)
    solved = [r for r in records if r["has_solutions"] and r["pass@1"] is not None]
    if not solved:
        print("  No solution data available for quality plots.")
        return

    tag_pieces = []
    if model_name:
        tag_pieces.append(model_name)
    if harness and harness != "bash":
        tag_pieces.append(harness)
    tag = f" [{' / '.join(tag_pieces)}]" if tag_pieces else ""
    all_domains = sorted({r["domain"] for r in records})

    # -- pass@1 charts --
    _bar_chart(
        solved, "domain", "pass@1", "Mean pass@1",
        f"Pass@1 by Domain{tag}", "quality_pass1_by_domain.png",
        out_dir, color="steelblue", expected_keys=all_domains,
    )
    _bar_chart(
        solved, "task_complexity", "pass@1", "Mean pass@1",
        f"Pass@1 by Task Complexity{tag}", "quality_pass1_by_task_complexity.png",
        out_dir, color="darkorange", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "command_complexity", "pass@1", "Mean pass@1",
        f"Pass@1 by Command Complexity{tag}", "quality_pass1_by_command_complexity.png",
        out_dir, color="mediumpurple", expected_keys=_CMD_COMPLEXITY_ORDER,
    )

    # -- pass@K (pass-at-any) charts — one set per K in the ladder --
    # Different K's get distinct colors so visually scanning the directory
    # makes it obvious which is which.
    _PK_COLORS_PRIMARY = {
        4: ("royalblue", "coral", "orchid"),  # (domain, task_cplx, cmd_cplx)
        8: ("midnightblue", "firebrick", "rebeccapurple"),
    }
    _PK_COLORS_V2 = {
        4: ("royalblue", "coral", "orchid", "peru"),  # 4 v2 axes
        8: ("midnightblue", "firebrick", "rebeccapurple", "saddlebrown"),
    }
    for K in pass_k_ladder:
        # Project pass@K for this iteration onto a stable record key. We
        # write a fresh field on each loop so _bar_chart can read by name.
        pk_field = f"_pass_at_k_{K}"
        for r in solved:
            r[pk_field] = _pass_at_k(r, K)
        pk_label = f"Pass@{K}"
        pk_fname = f"pass{K}"
        c_dom, c_tc, c_cc = _PK_COLORS_PRIMARY.get(
            K, ("royalblue", "coral", "orchid")
        )
        _bar_chart(
            solved, "domain", pk_field, f"Mean {pk_label.lower()}",
            f"{pk_label} by Domain{tag}",
            f"quality_{pk_fname}_by_domain.png",
            out_dir, color=c_dom, expected_keys=all_domains,
        )
        _bar_chart(
            solved, "task_complexity", pk_field, f"Mean {pk_label.lower()}",
            f"{pk_label} by Task Complexity{tag}",
            f"quality_{pk_fname}_by_task_complexity.png",
            out_dir, color=c_tc, expected_keys=_TASK_COMPLEXITY_ORDER,
        )
        _bar_chart(
            solved, "command_complexity", pk_field, f"Mean {pk_label.lower()}",
            f"{pk_label} by Command Complexity{tag}",
            f"quality_{pk_fname}_by_command_complexity.png",
            out_dir, color=c_cc, expected_keys=_CMD_COMPLEXITY_ORDER,
        )

    # -- turns charts --
    _bar_chart(
        solved, "task_complexity", "avg_turns", "Avg Turns",
        f"Average Turns by Task Complexity{tag}",
        "quality_turns_by_task_complexity.png",
        out_dir, color="seagreen", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "domain", "avg_turns", "Avg Turns",
        f"Average Turns by Domain{tag}", "quality_turns_by_domain.png",
        out_dir, color="teal", expected_keys=all_domains,
    )

    # -- v2-axis quality plots (only render when the axis varies in the
    #    solved set; otherwise a single bar at 100 % carries no info). ----
    def _seen_keys(field: str, order: List[str]) -> Optional[List[str]]:
        seen = {r[field] for r in solved}
        if len(seen) <= 1:
            return None
        return [k for k in order if k in seen] + sorted(seen - set(order))

    v2_axes = [
        ("verifier_kind", _VERIFIER_KIND_ORDER, "steelblue", "teal"),
        ("fixture_kind", _FIXTURE_KIND_ORDER, "darkorange", "seagreen"),
        ("corpus_kind", _CORPUS_KIND_ORDER, "mediumpurple", "darkslateblue"),
        ("base_image", _BASE_IMAGE_ORDER, "saddlebrown", "olive"),
    ]
    for field, order, color_p1, color_t in v2_axes:
        keys = _seen_keys(field, order)
        if keys is None:
            continue
        pretty = field.replace("_", " ").title()
        _bar_chart(
            solved, field, "pass@1", "Mean pass@1",
            f"Pass@1 by {pretty}{tag}", f"quality_pass1_by_{field}.png",
            out_dir, color=color_p1, expected_keys=keys,
        )
        _bar_chart(
            solved, field, "avg_turns", "Avg Turns",
            f"Average Turns by {pretty}{tag}", f"quality_turns_by_{field}.png",
            out_dir, color=color_t, expected_keys=keys,
        )
        for K_idx, K in enumerate(pass_k_ladder):
            pk_field = f"_pass_at_k_{K}"
            v2_palette = _PK_COLORS_V2.get(K, ("royalblue",) * 4)
            color_pk = v2_palette[
                {"verifier_kind": 0, "fixture_kind": 1,
                 "corpus_kind": 2, "base_image": 3}.get(field, 0)
            ]
            _bar_chart(
                solved, field, pk_field, f"Mean pass@{K}",
                f"Pass@{K} by {pretty}{tag}",
                f"quality_pass{K}_by_{field}.png",
                out_dir, color=color_pk, expected_keys=keys,
            )

    # --- Pass@k curve (averaged across tasks) ---
    # Build directly from records (which already cache pass_at_k_full) so we
    # don't re-read JSON. Plot every k present, but draw vertical guide
    # lines at the ladder K's to make ragged-k corpora visually obvious.
    # Each point is annotated with n = #tasks contributing at that k.
    all_pass_at_k: Dict[int, List[float]] = defaultdict(list)
    for r in solved:
        for k, v in (r.get("pass_at_k_full") or {}).items():
            all_pass_at_k[k].append(v)

    if all_pass_at_k:
        ks = sorted(all_pass_at_k.keys())
        means = [sum(all_pass_at_k[k]) / len(all_pass_at_k[k]) for k in ks]
        ns = [len(all_pass_at_k[k]) for k in ks]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, means, "o-", color="crimson", linewidth=2, markersize=5)
        for kk, m, nn in zip(ks, means, ns):
            ax.annotate(
                f"n={nn}", xy=(kk, m), xytext=(0, 6),
                textcoords="offset points", fontsize=7,
                ha="center", color="dimgray",
            )
        ax.set_xlabel("k")
        ax.set_ylabel("Mean pass@k")
        ax.set_title(
            f"Pass@k Curve (averaged across tasks){tag}", fontweight="bold",
        )
        ax.set_ylim(0, 1.10)
        ax.grid(True, alpha=0.3)
        for K in pass_k_ladder:
            if K in all_pass_at_k:
                ax.axvline(
                    K, color="gray", linestyle="--", alpha=0.5,
                    label=f"ladder k={K}",
                )
        if pass_k_ladder:
            ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_pass_at_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / 'quality_pass_at_k.png'}")

    # --- num_success distribution histogram ---
    # For each unique NUM_SOLUTIONS group, count how many tasks landed at
    # each num_success value 0..NUM_SOLUTIONS. This separates "easy" tasks
    # (large bar at N/N) from "hard" tasks (large bar at 0/N) at a glance —
    # complementary to the mean pass@k aggregates which collapse the shape.
    _plot_num_success_distribution(solved, out_dir, tag)


def _plot_num_success_distribution(
    solved: List[Dict[str, Any]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Render ``quality_num_success_distribution.png``.

    For each NUM_SOLUTIONS group (= unique ``num_runs`` value among solved
    records), emit a side-by-side bar chart of "tasks with X successes out
    of N" for X in 0..N. Ragged corpora (e.g. some tasks at NUM_SOLUTIONS=4,
    others at 8) get one subplot per group so the bin-width difference is
    explicit instead of papered over.
    """
    by_runs: Dict[int, List[int]] = defaultdict(list)
    for r in solved:
        n_runs = int(r.get("num_runs", 0))
        n_succ = int(r.get("num_success", 0))
        if n_runs <= 0:
            continue
        by_runs[n_runs].append(n_succ)
    if not by_runs:
        return

    groups = sorted(by_runs.items())  # [(num_runs, [success_counts])]
    n_groups = len(groups)
    fig, axes = plt.subplots(
        nrows=n_groups, ncols=1,
        figsize=(max(8, max(n_runs for n_runs, _ in groups) * 0.9),
                 3.5 * n_groups),
        squeeze=False,
    )
    for ax, (n_runs, successes) in zip(axes[:, 0], groups):
        counts = Counter(successes)
        xs = list(range(n_runs + 1))
        ys = [counts.get(x, 0) for x in xs]
        bars = ax.bar(
            xs, ys,
            color=["firebrick" if x == 0
                   else ("forestgreen" if x == n_runs else "steelblue")
                   for x in xs],
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}/{n_runs}" for x in xs], fontsize=9)
        ax.set_ylabel(f"# tasks (n={len(successes)})")
        ax.set_title(
            f"Solution Success Distribution — NUM_SOLUTIONS={n_runs}{tag}",
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)
        for bar, y in zip(bars, ys):
            if y == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(y), ha="center", va="bottom", fontsize=8,
            )
    fig.tight_layout()
    out_path = out_dir / "quality_num_success_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _analyze_model(
    tasks_dir: Path,
    plots_base: Path,
    model_slug: str,
    max_rows: int = 50,
    harness: str = "bash",
    pass_k_override: Optional[int] = None,
) -> None:
    """Run the full per-model analysis (table + quality plots).

    *harness* selects between bash (legacy filename ``<slug>_summary.json``)
    and any v2 harness (suffixed filename ``<slug>_<HARNESS>_summary.json``,
    e.g. ``vanillux``).

    *pass_k_override*: when ``None`` (default), the K-ladder is auto-resolved
    as every k in :data:`_DEFAULT_PASS_K_LADDER` for which ≥1 solved task
    has data (so a uniform NUM_SOLUTIONS=8 corpus reports both pass@4 AND
    pass@8). When set, the ladder collapses to ``[override]`` — useful for
    comparing two corpora at a fixed k.
    """
    display = model_slug.replace("_", "/", 1)

    records = load_tasks(tasks_dir, model_slug=model_slug, harness=harness)

    # Resolve the per-corpus pass@K ladder once. Fall back to [8] when
    # there are no solved tasks so column headers are still sensible.
    pk_ladder = _resolve_pass_k_ladder(records, override=pass_k_override)
    if not pk_ladder:
        pk_ladder = [8]

    # Keep bash output dirs at the legacy location; isolate non-bash harness
    # plots so a side-by-side bash+vanillux run on the same model produces
    # two cleanly-separated plot subtrees.
    sub = model_slug if harness == "bash" else f"{model_slug}__{harness}"
    model_dir = plots_base / sub
    print_summary(records, model_name=display, model_slug=model_slug,
                  max_rows=max_rows, harness=harness,
                  pass_k_ladder=pk_ladder)

    label = display if harness == "bash" else f"{display} [{harness}]"
    print(f"Generating quality plots for {label}...")
    plot_quality(records, model_dir, model_name=display,
                 model_slug=model_slug, harness=harness,
                 pass_k_ladder=pk_ladder)

    print(f"Done. Model plots saved to {model_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze generated RL tasks and solutions."
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory containing task_* subdirectories",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Where to save plots (default: <tasks-dir>/analysis)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model to analyze, e.g. 'gemini/gemini-3-flash-preview'. "
            "Omit to auto-discover and analyze all models."
        ),
    )
    ap.add_argument(
        "--harness",
        type=str,
        default=None,
        choices=("bash", "vanillux"),
        help=(
            "Solver harness to analyze. ``bash`` reads the legacy "
            "``<MODEL>_summary.json``; ``vanillux`` reads "
            "``<MODEL>_vanillux_summary.json`` (commit ad9d7fe). "
            "Omit to auto-discover all (model, harness) pairs."
        ),
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help=(
            "Max per-task rows to print (default 50). "
            "Set to 0 to show all rows regardless of count."
        ),
    )
    ap.add_argument(
        "--pass-at-k",
        type=int,
        default=None,
        help=(
            "Force the K-ladder to a single value. Default: auto-detect — "
            "report every k in (4, 8) for which at least one solved task "
            "has data. So a uniform NUM_SOLUTIONS=8 corpus shows both "
            "pass@4 AND pass@8 columns/plots; a NUM_SOLUTIONS=4 smoke "
            "shows only pass@4. Tasks without a value at a given k are "
            "excluded from that cell only (n-counts are surfaced)."
        ),
    )
    args = ap.parse_args()

    tasks_dir = args.tasks_dir
    plots_dir = args.plots_dir or (tasks_dir / "analysis")

    print(f"Scanning {tasks_dir}...")

    # Distribution plots use all tasks (model-independent)
    all_records = load_tasks(tasks_dir)
    if not all_records:
        print("No tasks found.")
        return

    print("Generating distribution plots...")
    plot_distributions(all_records, plots_dir)

    # Determine which (model, harness) pair(s) to analyze
    if args.model:
        slug = args.model.replace("/", "_")
        if args.harness:
            pairs = [(slug, args.harness)]
        else:
            # If a model is given but the harness isn't, prefer bash for
            # backwards compatibility but also analyze any non-bash harness
            # summaries that happen to exist on disk.
            pairs = [(slug, "bash")]
            for h in _KNOWN_HARNESSES:
                # Cheap probe — does ANY task have a summary for this harness?
                probe = list(tasks_dir.glob(
                    f"*/solutions/{slug}_{h}_summary.json"
                ))
                if probe:
                    pairs.append((slug, h))
    else:
        pairs = discover_models(tasks_dir)
        if args.harness:
            pairs = [(s, h) for (s, h) in pairs if h == args.harness]
        if not pairs:
            print("No model summaries found — nothing to analyze.")
            return
        pretty = ", ".join(
            s if h == "bash" else f"{s} [{h}]" for s, h in pairs
        )
        print(f"Discovered {len(pairs)} (model, harness) pair(s): {pretty}")

    for slug, harness in pairs:
        _analyze_model(
            tasks_dir, plots_dir, slug,
            max_rows=args.max_rows, harness=harness,
            pass_k_override=args.pass_at_k,
        )

    print(f"\nDone. All plots saved under {plots_dir}/")


if __name__ == "__main__":
    main()
