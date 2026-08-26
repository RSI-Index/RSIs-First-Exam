#!/usr/bin/env python3
"""Render the measured Step 4 training and checkpoint-eval reward curves."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAIN_CSV = HERE / "STEP4_BASELINE_TRAIN_REWARD.csv"
EVAL_CSV = HERE / "STEP4_BASELINE_CHECKPOINT_EVAL.csv"
OUTPUT = HERE / "STEP4_BASELINE_REWARD_CURVES.svg"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def polyline(points: list[tuple[float, float]], color: str, width: float) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def main() -> None:
    train = read_rows(TRAIN_CSV)
    checkpoint_eval = read_rows(EVAL_CSV)

    width, height = 1000, 720
    left, right = 78, 35
    plot_width = width - left - right
    top1, panel_height = 70, 255
    top2 = 420

    def x_train(update: float) -> float:
        return left + (update - 1) / 59 * plot_width

    def x_eval(step: float) -> float:
        return left + (step - 20) / 40 * plot_width

    def y(value: float, top: float) -> float:
        return top + (1.0 - value) * panel_height

    train_points = [
        (x_train(float(row["updates_completed"])), y(float(row["reward"]), top1))
        for row in train
    ]
    rewards = [float(row["reward"]) for row in train]
    rolling_points: list[tuple[float, float]] = []
    for index, value in enumerate(rewards):
        start = max(0, index - 4)
        rolling = sum(rewards[start : index + 1]) / (index - start + 1)
        rolling_points.append((x_train(index + 1), y(rolling, top1)))

    eval_points = [
        (x_eval(float(row["checkpoint"])), y(float(row["raw_oolong_reward"]), top2))
        for row in checkpoint_eval
    ]

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{font-size:13px}.title{font-size:20px;font-weight:700}.subtitle{font-size:15px;font-weight:700}</style>',
        '<text x="500" y="30" text-anchor="middle" class="title">RLM Step 4 baseline reward curves</text>',
        f'<text x="{left}" y="55" class="subtitle">Training reward: OOLONG spam rollouts</text>',
        f'<text x="{left}" y="405" class="subtitle">Checkpoint eval reward: OOLONG trec_coarse, 25 rows</text>',
    ]

    for top in (top1, top2):
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y(value, top)
            out.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" class="grid"/>')
            out.append(f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" class="label">{value:.2f}</text>')
        out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_height}" class="axis"/>')
        out.append(f'<line x1="{left}" y1="{top+panel_height}" x2="{width-right}" y2="{top+panel_height}" class="axis"/>')

    for update in (1, 10, 20, 30, 40, 50, 60):
        xx = x_train(update)
        out.append(f'<text x="{xx:.2f}" y="{top1+panel_height+22}" text-anchor="middle" class="label">{update}</text>')
    for step in (20, 40, 60):
        xx = x_eval(step)
        out.append(f'<line x1="{xx:.2f}" y1="{top1}" x2="{xx:.2f}" y2="{top1+panel_height}" stroke="#bbb" stroke-dasharray="4 4"/>')
        out.append(f'<text x="{xx:.2f}" y="{top2+panel_height+22}" text-anchor="middle" class="label">{step}</text>')

    out.append(polyline(train_points, "#9aa0a6", 1.5))
    out.append(polyline(rolling_points, "#e8710a", 3.0))
    out.append(f'<text x="{width-right-160}" y="{top1+20}" class="label" fill="#e8710a">orange: trailing 5-step mean</text>')
    out.append(polyline(eval_points, "#1a73e8", 3.0))
    for (xx, yy), row in zip(eval_points, checkpoint_eval):
        reward = float(row["raw_oolong_reward"])
        out.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="6" fill="#1a73e8"/>')
        out.append(f'<text x="{xx:.2f}" y="{yy-12:.2f}" text-anchor="middle" class="label">{reward:.6f}</text>')

    out.extend(
        [
            f'<text x="{width/2}" y="{top1+panel_height+42}" text-anchor="middle" class="label">Updates completed</text>',
            f'<text x="{width/2}" y="{top2+panel_height+42}" text-anchor="middle" class="label">Checkpoint step</text>',
            '</svg>',
        ]
    )
    OUTPUT.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
