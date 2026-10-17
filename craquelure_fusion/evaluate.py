"""The experiment. Run:
    python -m craquelure_fusion.evaluate

Protocol (decided before looking at eval numbers):
  1. For each model, choose the cutoff on the VALIDATION set: the strictest cutoff
     that still finds TARGET_RECALL of crack pixels. Same recall for every model,
     so the models are compared like for like.
  2. Apply that cutoff to the EVAL set. Report per sample: precision, recall,
     stroke false alarms, impasto false alarms, and average precision (AP).
  3. Paired comparison, fused vs each single sensor, sample by sample:
     mean difference with a 95% bootstrap interval, and a Wilcoxon signed-rank test.
  4. Stress test: shift the height map 0-4 px against the photo (misregistration)
     and see how fast the fused model degrades.

Outputs in results/: per_sample.csv, summary.json, pr_curves.png, misregistration.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import stats

from craquelure_fusion.baseline import MODELS, score_fused
from craquelure_fusion.generate import Sample, load_sample
from craquelure_fusion.metrics import false_alarm_rate, tolerant_pr_curve, tolerant_precision_recall

logger = logging.getLogger(__name__)

TARGET_RECALL = 0.90
FUSED = "colour_plus_height"
SHIFTS_PX = (0, 1, 2, 3, 4)


def load_split(split_dir: Path) -> list[tuple[str, Sample]]:
    dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())
    return [(d.name, load_sample(d)) for d in dirs]


def choose_thresholds(val: list[tuple[str, Sample]]) -> dict[str, float]:
    truths = [s.crack_mask for _, s in val]
    return {
        name: tolerant_pr_curve([f(s) for _, s in val], truths).threshold_at_recall(TARGET_RECALL)
        for name, f in MODELS.items()
    }


def per_sample_rows(eval_set: list[tuple[str, Sample]], thresholds: dict[str, float]) -> list[dict]:
    rows = []
    for sample_id, s in eval_set:
        for name, f in MODELS.items():
            score = f(s)
            pred = score >= thresholds[name]
            precision, recall = tolerant_precision_recall(pred, s.crack_mask)
            rows.append({
                "sample": sample_id,
                "model": name,
                "ap": round(tolerant_pr_curve([score], [s.crack_mask]).average_precision, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "stroke_false_alarm": round(false_alarm_rate(pred, s.crack_mask, s.stroke_mask), 4),
                "impasto_false_alarm": round(false_alarm_rate(pred, s.crack_mask, s.impasto_mask), 4),
            })
    return rows


def paired_comparison(rows: list[dict], a: str, b: str, metric: str) -> dict:
    """b minus a, sample by sample. Same samples, so the noise between samples cancels."""
    va = np.array([r[metric] for r in rows if r["model"] == a])
    vb = np.array([r[metric] for r in rows if r["model"] == b])
    diff = vb - va
    out = {"mean_difference": float(diff.mean())}
    if np.allclose(diff, diff[0]):  # no spread: interval and test are meaningless
        out.update(ci95=[float(diff[0]), float(diff[0])], wilcoxon_p=None)
        return out
    ci = stats.bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=5000,
                         method="percentile", random_state=0).confidence_interval
    out.update(ci95=[float(ci.low), float(ci.high)], wilcoxon_p=float(stats.wilcoxon(diff).pvalue))
    return out


def misregistration(eval_set: list[tuple[str, Sample]]) -> dict[int, float]:
    """Fused AP when the height map is shifted right by k px (the photo stays put)."""
    truths = [s.crack_mask for _, s in eval_set]
    result = {}
    for k in SHIFTS_PX:
        shifted = [replace(s, height=np.roll(s.height, k, axis=1)) for _, s in eval_set]
        result[k] = tolerant_pr_curve([score_fused(s) for s in shifted], truths).average_precision
    return result


def save_plots(eval_set, mis: dict[int, float], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truths = [s.crack_mask for _, s in eval_set]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, f in MODELS.items():
        c = tolerant_pr_curve([f(s) for _, s in eval_set], truths)
        ax.plot(c.recall, c.precision, label=f"{name} (AP {c.average_precision:.2f})")
    ax.axvline(TARGET_RECALL, color="grey", ls=":", lw=1)
    ax.set(xlabel="recall (cracks found)", ylabel="precision (predictions that are cracks)",
           xlim=(0, 1.02), ylim=(0, 1.02), title="Eval set v2, tolerance 2 px")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(list(mis), list(mis.values()), marker="o")
    ax.set(xlabel="height map shifted by (px)", ylabel="fused AP", ylim=(0, 1.02),
           title="How alignment limits fusion")
    fig.tight_layout()
    fig.savefig(out_dir / "misregistration.png", dpi=130)
    plt.close(fig)


def run(val_dir: Path, eval_dir: Path, out_dir: Path) -> dict:
    val, eval_set = load_split(val_dir), load_split(eval_dir)
    thresholds = choose_thresholds(val)
    rows = per_sample_rows(eval_set, thresholds)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = ("ap", "precision", "recall", "stroke_false_alarm", "impasto_false_alarm")
    mis = misregistration(eval_set)
    summary = {
        "protocol": {"val": str(val_dir), "eval": str(eval_dir), "target_recall": TARGET_RECALL,
                     "tolerance_px": 2, "n_eval_samples": len(eval_set)},
        "thresholds_from_val": thresholds,
        "means": {m: {k: float(np.mean([r[k] for r in rows if r["model"] == m])) for k in metrics}
                  for m in MODELS},
        "fused_vs": {
            other: {k: paired_comparison(rows, other, FUSED, k)
                    for k in ("ap", "stroke_false_alarm", "impasto_false_alarm")}
            for other in MODELS if other != FUSED
        },
        "misregistration_fused_ap": {str(k): v for k, v in mis.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_plots(eval_set, mis, out_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the experiment on the fixed sets.")
    parser.add_argument("--val", type=Path, default=Path("data/val_v2"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval_v2"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    s = run(args.val, args.eval, args.out)
    print(f"cutoffs chosen on {args.val} at recall {TARGET_RECALL}; scored on {args.eval}\n")
    print(f"{'model':<20}{'AP':>7}{'prec':>7}{'recall':>8}{'stroke FA':>11}{'impasto FA':>12}")
    for m, v in s["means"].items():
        print(f"{m:<20}{v['ap']:>7.3f}{v['precision']:>7.3f}{v['recall']:>8.3f}"
              f"{v['stroke_false_alarm']:>11.3f}{v['impasto_false_alarm']:>12.3f}")
    print("\nfused minus single sensor (mean, 95% CI, Wilcoxon p):")
    for other, comp in s["fused_vs"].items():
        for k, c in comp.items():
            p = "n/a" if c["wilcoxon_p"] is None else f"{c['wilcoxon_p']:.1e}"
            print(f"  vs {other:<12} {k:<20} {c['mean_difference']:+.3f}  "
                  f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  p={p}")
    print("\nfused AP vs height-map shift:",
          ", ".join(f"{k}px {v:.2f}" for k, v in s["misregistration_fused_ap"].items()))
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
