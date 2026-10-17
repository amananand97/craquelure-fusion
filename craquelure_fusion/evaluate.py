"""Score every model on the FIXED evaluation set and write one results table.

Reads the saved files from disk (data/eval/), never from memory, so the score
is exactly what anyone can reproduce from the committed data.

Run:
    python -m craquelure_fusion.evaluate --eval data/eval --out results/results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from craquelure_fusion.baseline import predict_colour_only, predict_fused
from craquelure_fusion.generate import load_sample
from craquelure_fusion.metrics import stroke_false_alarm_rate, tolerant_precision_recall

logger = logging.getLogger(__name__)

MODELS = {
    "A_colour_only": lambda s: predict_colour_only(s.rgb),
    "B_colour_plus_height": lambda s: predict_fused(s.rgb, s.height),
}


def evaluate(eval_dir: Path) -> list[dict]:
    """One row per (sample, model): precision, recall, stroke false-alarm rate."""
    rows = []
    for sample_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        sample = load_sample(sample_dir)
        for name, predict in MODELS.items():
            pred = predict(sample)
            precision, recall = tolerant_precision_recall(pred, sample.crack_mask)
            rows.append({
                "sample": sample_dir.name,
                "model": name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "stroke_false_alarm": round(stroke_false_alarm_rate(pred, sample.crack_mask, sample.stroke_mask), 4),
            })
    return rows


def summarise(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean of each score per model."""
    summary = {}
    for name in MODELS:
        mine = [r for r in rows if r["model"] == name]
        summary[name] = {
            key: float(np.mean([r[key] for r in mine]))
            for key in ("precision", "recall", "stroke_false_alarm")
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate all models on the fixed eval set.")
    parser.add_argument("--eval", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, default=Path("results/results.csv"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    rows = evaluate(args.eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'model':<24}{'precision':>10}{'recall':>10}{'stroke FA':>12}")
    for name, s in summarise(rows).items():
        print(f"{name:<24}{s['precision']:>10.3f}{s['recall']:>10.3f}{s['stroke_false_alarm']:>12.3f}")
    print(f"per-sample rows written to {args.out}")


if __name__ == "__main__":
    main()
