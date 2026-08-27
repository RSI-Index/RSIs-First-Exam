"""Compute avg 'peak context' stats per run for skill_tax 10k.

For each run:
  - The recorded per-run API usage is a SUM across turns:
        prompt_tokens     = sum_{t=1..T} (prompt sent on turn t)
        completion_tokens = sum_{t=1..T} (assistant reply on turn t)
    where each turn's prompt includes the full conversation so far.

  - We estimate the tokens-per-word conversion factor by equating:
        prompt_tokens = tpw * [ T*(sys+user)_w
                                + sum_{i=1..T-1} (T-i)*(asst_i + tool_i)_w ]
        completion_tokens = tpw * sum_{i=1..T} asst_i_w
    and taking the ratio that best fits both.

  - Final-turn input tokens = tpw * (all messages before last assistant)_w
  - Initial input tokens    = tpw * (system + first user)_w

This way, summing per-turn inputs across all T turns exactly reproduces the
recorded `prompt_tokens`, and the final-turn input is bounded above by that
sum as expected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

TASKS_DIR = Path("/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_skill_tax_20260401_10k")
MODEL = "gemini_gemini-3-flash-preview"


def _words(msg: Dict[str, Any]) -> int:
    content = msg.get("content") or ""
    w = len(content.split()) if content else 0
    for tc in (msg.get("tool_calls") or []):
        args = tc.get("function", {}).get("arguments", "")
        if args:
            w += len(args.split())
    return w


def _process_run(result: Dict[str, Any]) -> Dict[str, float] | None:
    msgs = result.get("messages", [])
    if not msgs:
        return None

    sys_user_w = 0
    i = 0
    while i < len(msgs) and msgs[i].get("role") in ("system", "user"):
        sys_user_w += _words(msgs[i])
        i += 1
    first_turn_input_w = sys_user_w

    turn_pair_words: List[int] = []
    current_asst_w = 0
    current_tool_w = 0
    saw_asst = False
    for m in msgs[i:]:
        role = m.get("role")
        w = _words(m)
        if role == "assistant":
            if saw_asst:
                turn_pair_words.append(current_asst_w + current_tool_w)
                current_asst_w = 0
                current_tool_w = 0
            current_asst_w = w
            saw_asst = True
        elif role == "tool":
            current_tool_w += w
        else:
            if saw_asst:
                current_tool_w += w
    if saw_asst:
        turn_pair_words.append(current_asst_w + current_tool_w)

    T = len(turn_pair_words)
    if T == 0:
        return None

    prompt_sum_w = T * sys_user_w
    for j in range(T - 1):
        prompt_sum_w += (T - 1 - j) * turn_pair_words[j]

    completion_sum_w = 0
    running_asst_w = 0
    saw_asst = False
    for m in msgs[i:]:
        role = m.get("role")
        w = _words(m)
        if role == "assistant":
            completion_sum_w += w

    usage = result.get("usage") or {}
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    if prompt_sum_w + completion_sum_w <= 0 or pt + ct <= 0:
        return None
    tpw = (pt + ct) / (prompt_sum_w + completion_sum_w)

    pre_last_asst_w = sys_user_w + sum(turn_pair_words[: T - 1])
    last_asst_w = 0
    saw = 0
    for m in msgs[i:]:
        if m.get("role") == "assistant":
            saw += 1
            if saw == T:
                last_asst_w = _words(m)
                break

    final_in_tok = pre_last_asst_w * tpw
    final_out_tok = last_asst_w * tpw
    initial_in_tok = sys_user_w * tpw
    first_turn_input_tok = first_turn_input_w * tpw

    return {
        "final_in_tok": final_in_tok,
        "final_out_tok": final_out_tok,
        "initial_in_tok": initial_in_tok,
        "first_turn_input_tok": first_turn_input_tok,
        "per_run_total_in_tok": pt,
        "per_run_total_out_tok": ct,
        "num_turns": T,
    }


def main() -> None:
    task_dirs = [p for p in sorted(TASKS_DIR.iterdir())
                 if p.is_dir() and p.name.startswith("task_")]
    print(f"Scanning {len(task_dirs)} task dirs...")

    n_tasks = 0
    n_runs = 0
    sums: Dict[str, float] = {
        "final_in_tok": 0.0,
        "final_out_tok": 0.0,
        "initial_in_tok": 0.0,
        "first_turn_input_tok": 0.0,
        "per_run_total_in_tok": 0.0,
        "per_run_total_out_tok": 0.0,
        "num_turns": 0.0,
    }
    final_in_list: List[float] = []

    for idx, tdir in enumerate(task_dirs):
        sp = tdir / "solutions" / f"{MODEL}_summary.json"
        if not sp.exists():
            continue
        try:
            with open(sp) as f:
                sol = json.load(f)
        except Exception:
            continue
        has_any_run = False
        for r in sol.get("results", []):
            stats = _process_run(r)
            if not stats:
                continue
            has_any_run = True
            n_runs += 1
            for k, v in stats.items():
                sums[k] += v
            final_in_list.append(stats["final_in_tok"])
        if has_any_run:
            n_tasks += 1
        if (idx + 1) % 2000 == 0:
            print(f"  processed {idx+1} tasks, {n_runs} runs so far")

    print()
    print(f"Tasks with solutions processed: {n_tasks}")
    print(f"Total runs processed:           {n_runs}")
    if n_runs == 0:
        return

    avg = {k: v / n_runs for k, v in sums.items()}
    print()
    print(f"Avg turns per run: {avg['num_turns']:.2f}")
    print()
    print("Per-run averages (tokens):")
    print(f"  initial input (system+first user)  : {avg['initial_in_tok']:>10,.0f}")
    print(f"  final-turn input  (peak context)   : {avg['final_in_tok']:>10,.0f}")
    print(f"  final-turn output                  : {avg['final_out_tok']:>10,.0f}")
    print(f"  peak context − initial input       : "
          f"{avg['final_in_tok']-avg['initial_in_tok']:>10,.0f}")
    print()
    print("Sanity check (recorded API per-run totals, summed over all turns):")
    print(f"  total input  over all turns        : {avg['per_run_total_in_tok']:>10,.0f}")
    print(f"  total output over all turns        : {avg['per_run_total_out_tok']:>10,.0f}")

    if final_in_list:
        final_in_list.sort()
        def pct(p: float) -> float:
            k = int(round(p * (len(final_in_list) - 1)))
            return final_in_list[k]
        print()
        print("Final-turn input tokens distribution across runs:")
        print(f"  p50 : {pct(0.50):>10,.0f}")
        print(f"  p90 : {pct(0.90):>10,.0f}")
        print(f"  p95 : {pct(0.95):>10,.0f}")
        print(f"  p99 : {pct(0.99):>10,.0f}")
        print(f"  max : {max(final_in_list):>10,.0f}")


if __name__ == "__main__":
    main()
