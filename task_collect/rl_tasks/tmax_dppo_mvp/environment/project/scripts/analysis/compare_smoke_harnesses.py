"""Compare bash vs vanillux harness rollouts on a v2 smoke run.

Usage
-----
    python scripts/analysis/compare_smoke_harnesses.py \\
        rl_data/output/tasks_skill_tax_v2_20260505_2k \\
        --model hosted_vllm/Qwen/Qwen3.6-27B \\
        [--sample-size 25 --sample-seed 0]

What it reports
---------------
1. Aggregate pass@1 / pass@k for both harnesses, restricted to the
   intersection of tasks that have BOTH summaries (apples-to-apples).
2. Per-task wins / losses / ties.
3. Per-harness diagnostics from each trial's message thread:
     * num_steps (assistant turns observed)
     * exit reason: submitted / max_steps / timeout / no_tool_call /
       format_error / vllm_error / other
     * avg / max input prompt size in chars (proxy for context pressure)
4. Token-usage delta (prompt + completion) per harness.
5. Top-5 failing tasks for vanillux only (where bash succeeded), with
   exit-reason breakdown — these are the cases worth eyeballing.

The script is intended to be reusable across (model, corpus) pairs:
just point at a different corpus dir + --model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _model_tag(model: str) -> str:
    return model.replace("/", "_")


def _summary_filename(model_tag: str, harness: str) -> str:
    """Mirrors rl_data.generate_solutions._summary_basename."""
    if harness == "bash":
        return f"{model_tag}_summary.json"
    return f"{model_tag}_{harness}_summary.json"


def _sample_task_dirs(corpus_dir: Path, sample_size: int, sample_seed: int) -> List[Path]:
    """Reproduce the sampling that the smoke script uses."""
    all_dirs = sorted(d for d in corpus_dir.glob("task_*") if d.is_dir())
    if sample_size and sample_size < len(all_dirs):
        rng = random.Random(sample_seed)
        sampled = rng.sample(all_dirs, sample_size)
        return sorted(sampled)
    return all_dirs


def _classify_trial(trial: Dict[str, Any]) -> Tuple[str, int]:
    """Return (exit_reason, num_assistant_turns) for one trial.

    Exit reasons (rough taxonomy):
        submitted          - last command echoed COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
        no_submit          - reached end of trajectory without echoing the marker
        timeout            - 'Command timed out' detected in a tool result
        max_steps          - presence of explicit "hit max_actions" message
        format_error       - any 'Format error' / 'Your last response' assistant
                             reprompt (mini-swe-agent style format-error recovery)
        vllm_error         - any tool-result containing a 4xx/5xx HTTP-style error
        other              - none of the above
    """
    messages: List[Dict[str, Any]] = trial.get("messages", []) or []

    submitted = False
    saw_timeout = False
    saw_format_error = False
    saw_vllm_error = False
    n_assistant = 0
    last_tool_content = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "assistant":
            n_assistant += 1
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                args_raw = fn.get("arguments", "")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                cmd = (args or {}).get("command", "") if isinstance(args, dict) else ""
                if SUBMIT_MARKER in cmd:
                    submitted = True
        elif role == "tool":
            last_tool_content = content
            if SUBMIT_MARKER in content:
                submitted = True
            if "Command timed out" in content:
                saw_timeout = True
            if re.search(r"\b(429|500|502|503|504)\b", content) and "Bad Request" in content:
                saw_vllm_error = True
            if "BadRequestError" in content or "ContextWindowExceeded" in content:
                saw_vllm_error = True
        elif role == "user":
            # mini-swe-agent format-error recovery emits a `user` message
            # starting with "Format error:" — see vanillux_solver.py.
            if content.startswith("Format error:") or "Please always provide EXACTLY ONE" in content:
                saw_format_error = True

    if submitted:
        return ("submitted", n_assistant)
    if saw_vllm_error:
        return ("vllm_error", n_assistant)
    if saw_timeout:
        return ("timeout", n_assistant)
    if saw_format_error:
        return ("format_error", n_assistant)
    return ("no_submit", n_assistant)


def _input_chars_stats(trial: Dict[str, Any]) -> Tuple[int, int]:
    """Return (max_input_chars_seen, total_chars_at_end).

    A rough proxy for context pressure; we accumulate all message contents.
    """
    messages = trial.get("messages", []) or []
    total = 0
    running_max = 0
    running_total = 0
    for msg in messages:
        c = msg.get("content", "") or ""
        if isinstance(c, list):  # multimodal-style
            c = json.dumps(c)
        chunk = len(c)
        running_total += chunk
        if running_total > running_max:
            running_max = running_total
        total = running_total
    return running_max, total


def _load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def analyse(
    corpus_dir: Path,
    model: str,
    harnesses: List[str],
    sample_size: int,
    sample_seed: int,
) -> None:
    model_tag = _model_tag(model)
    sampled = _sample_task_dirs(corpus_dir, sample_size, sample_seed)
    print(f"Corpus       : {corpus_dir}")
    print(f"Model        : {model}")
    print(f"Harnesses    : {harnesses}")
    print(f"Sample       : size={len(sampled)} seed={sample_seed}")
    print()

    # Step 1: aggregate per-harness, intersected on tasks with summaries
    #         from BOTH harnesses (apples-to-apples).
    per_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for d in sampled:
        per_task[d.name] = {}
        for h in harnesses:
            p = d / "solutions" / _summary_filename(model_tag, h)
            s = _load_summary(p)
            if s is not None:
                per_task[d.name][h] = s

    intersect_tasks = [
        name
        for name, by_h in per_task.items()
        if all(h in by_h for h in harnesses)
    ]
    print(f"Tasks in intersection (have all {harnesses} summaries): "
          f"{len(intersect_tasks)} / {len(sampled)}")
    print()

    if not intersect_tasks:
        print("Nothing to compare. Exiting.")
        return

    # Step 2: aggregate pass rates
    print("=" * 70)
    print("AGGREGATE PASS RATES (intersected tasks only)")
    print("=" * 70)
    print(f"{'harness':<10}  {'tasks':>6}  {'mean p@1':>10}  {'mean p@k':>10}  "
          f"{'p@k>0':>10}  {'k':>3}")
    agg_rows: Dict[str, Dict[str, Any]] = {}
    for h in harnesses:
        n_eval = 0
        sum_p1 = 0.0
        sum_pk = 0.0
        solved_some = 0
        ks = set()
        for name in intersect_tasks:
            s = per_task[name][h]
            n = s.get("num_runs", 0)
            c = s.get("num_success", 0)
            if n == 0:
                continue
            ks.add(n)
            n_eval += 1
            sum_p1 += c / n
            sum_pk += 1.0 if c >= n else (1.0 - math.comb(n - c, n) / math.comb(n, n))
            if c > 0:
                solved_some += 1
        k_label = max(ks) if ks else 0
        print(f"{h:<10}  {n_eval:>6}  "
              f"{(sum_p1 / n_eval if n_eval else 0):>10.3f}  "
              f"{(sum_pk / n_eval if n_eval else 0):>10.3f}  "
              f"{solved_some:>4} / {n_eval:<3}  {k_label:>3}")
        agg_rows[h] = {
            "n_eval": n_eval,
            "p1": sum_p1 / n_eval if n_eval else 0,
            "pk": sum_pk / n_eval if n_eval else 0,
            "solved_some": solved_some,
            "k": k_label,
        }

    # Step 3: per-task win/loss matrix (only meaningful for exactly 2 harnesses)
    if len(harnesses) == 2:
        h_a, h_b = harnesses
        print()
        print("=" * 70)
        print(f"PER-TASK: {h_a} vs {h_b} (num_success out of num_runs)")
        print("=" * 70)
        a_better = b_better = tie = 0
        rows: List[Tuple[str, int, int, int, int, str]] = []
        for name in intersect_tasks:
            sa = per_task[name][h_a]
            sb = per_task[name][h_b]
            ca, na = sa.get("num_success", 0), sa.get("num_runs", 0)
            cb, nb = sb.get("num_success", 0), sb.get("num_runs", 0)
            verdict = "= "
            if ca > cb:
                a_better += 1
                verdict = f"{h_a:<8} +"
            elif cb > ca:
                b_better += 1
                verdict = f"{h_b:<8} +"
            else:
                tie += 1
            rows.append((name, ca, na, cb, nb, verdict))
        for name, ca, na, cb, nb, verdict in rows:
            print(f"  {name:<32}  {h_a}: {ca}/{na}   {h_b}: {cb}/{nb}   {verdict}")
        print()
        print(f"  {h_a} better : {a_better}")
        print(f"  {h_b} better : {b_better}")
        print(f"  ties       : {tie}")

    # Step 4: per-harness exit-reason breakdown across all trials
    print()
    print("=" * 70)
    print("EXIT REASONS (per trial, across all intersected tasks)")
    print("=" * 70)
    exit_breakdown: Dict[str, Counter] = {}
    turn_stats: Dict[str, List[int]] = {}
    ctx_stats: Dict[str, List[int]] = {}
    success_turns: Dict[str, List[int]] = {}
    fail_turns: Dict[str, List[int]] = {}
    for h in harnesses:
        exits: Counter = Counter()
        all_turns: List[int] = []
        all_ctx: List[int] = []
        s_turns: List[int] = []
        f_turns: List[int] = []
        for name in intersect_tasks:
            s = per_task[name][h]
            for trial in s.get("results", []):
                reason, n_assistant = _classify_trial(trial)
                exits[reason] += 1
                all_turns.append(n_assistant)
                if trial.get("success"):
                    s_turns.append(n_assistant)
                else:
                    f_turns.append(n_assistant)
                max_chars, _ = _input_chars_stats(trial)
                all_ctx.append(max_chars)
        exit_breakdown[h] = exits
        turn_stats[h] = all_turns
        ctx_stats[h] = all_ctx
        success_turns[h] = s_turns
        fail_turns[h] = f_turns

    print(f"{'harness':<10}  " + "  ".join(f"{r:>14}" for r in
        ("submitted", "no_submit", "timeout", "format_error", "vllm_error", "max_steps")))
    for h in harnesses:
        ex = exit_breakdown[h]
        cells = [
            ex.get("submitted", 0),
            ex.get("no_submit", 0),
            ex.get("timeout", 0),
            ex.get("format_error", 0),
            ex.get("vllm_error", 0),
            ex.get("max_steps", 0),
        ]
        print(f"{h:<10}  " + "  ".join(f"{c:>14}" for c in cells))

    # Step 5: turn / context stats
    print()
    print("=" * 70)
    print("TURN COUNT & CONTEXT PRESSURE")
    print("=" * 70)
    print(f"{'harness':<10}  {'mean_turns':>10}  {'med':>5}  {'max':>5}  "
          f"{'mean_ok':>8}  {'mean_ko':>8}  {'mean_ctx':>10}  {'max_ctx':>10}")
    for h in harnesses:
        ts = turn_stats[h]
        cs = ctx_stats[h]
        ok = success_turns[h]
        ko = fail_turns[h]
        if not ts:
            continue
        print(
            f"{h:<10}  "
            f"{statistics.mean(ts):>10.1f}  "
            f"{statistics.median(ts):>5.0f}  "
            f"{max(ts):>5d}  "
            f"{(statistics.mean(ok) if ok else 0):>8.1f}  "
            f"{(statistics.mean(ko) if ko else 0):>8.1f}  "
            f"{(statistics.mean(cs) if cs else 0):>10.0f}  "
            f"{(max(cs) if cs else 0):>10d}"
        )

    # Step 6: token usage
    print()
    print("=" * 70)
    print("TOKEN USAGE TOTALS (sum across all tasks × all trials)")
    print("=" * 70)
    print(f"{'harness':<10}  {'prompt_tok':>14}  {'completion_tok':>16}  "
          f"{'total_tok':>14}  {'avg_total/trial':>18}")
    for h in harnesses:
        p_tot = c_tot = t_tot = 0
        n_trials = 0
        for name in intersect_tasks:
            s = per_task[name][h]
            for trial in s.get("results", []):
                u = trial.get("usage") or {}
                p_tot += u.get("prompt_tokens", 0) or 0
                c_tot += u.get("completion_tokens", 0) or 0
                t_tot += u.get("total_tokens", 0) or 0
                n_trials += 1
        avg = (t_tot / n_trials) if n_trials else 0
        print(f"{h:<10}  {p_tot:>14,}  {c_tot:>16,}  {t_tot:>14,}  {avg:>18,.0f}")

    # Step 7: tasks where ONE harness succeeded but the OTHER didn't (interesting cases)
    if len(harnesses) == 2:
        h_a, h_b = harnesses
        print()
        print("=" * 70)
        print(f"INTERESTING: tasks where ONLY ONE harness solved the problem")
        print("=" * 70)
        only_a: List[str] = []
        only_b: List[str] = []
        for name in intersect_tasks:
            ca = per_task[name][h_a].get("num_success", 0)
            cb = per_task[name][h_b].get("num_success", 0)
            if ca > 0 and cb == 0:
                only_a.append(name)
            elif cb > 0 and ca == 0:
                only_b.append(name)
        print(f"Only {h_a:<10} solved : {len(only_a)} tasks")
        for name in only_a:
            print(f"  {name}  ({h_a}: {per_task[name][h_a]['num_success']}/{per_task[name][h_a]['num_runs']})")
        print(f"Only {h_b:<10} solved : {len(only_b)} tasks")
        for name in only_b:
            print(f"  {name}  ({h_b}: {per_task[name][h_b]['num_success']}/{per_task[name][h_b]['num_runs']})")

        # For tasks where vanillux failed on every attempt, dump the
        # exit-reason breakdown across its 4 attempts.
        h_vlx = "vanillux" if "vanillux" in harnesses else h_b
        print()
        print(f"FAILURE MODES on tasks where {h_vlx} got 0/{4}:")
        for name in intersect_tasks:
            sv = per_task[name][h_vlx]
            if sv.get("num_success", 0) > 0:
                continue
            reasons: Counter = Counter()
            for trial in sv.get("results", []):
                reason, _ = _classify_trial(trial)
                reasons[reason] += 1
            top = ", ".join(f"{r}={n}" for r, n in reasons.most_common())
            print(f"  {name}: {top}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--model", default="hosted_vllm/Qwen/Qwen3.6-27B")
    ap.add_argument("--harnesses", nargs="+", default=["bash", "vanillux"],
                    help="One or more harness ids to compare.")
    ap.add_argument("--sample-size", type=int, default=25,
                    help="Reproduce the smoke sampling. 0 = use all tasks under corpus_dir.")
    ap.add_argument("--sample-seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyse(
        corpus_dir=args.corpus_dir.resolve(),
        model=args.model,
        harnesses=list(args.harnesses),
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
