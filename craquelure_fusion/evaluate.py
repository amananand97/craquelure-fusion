"""The experiment. Run:
    python -m craquelure_fusion.evaluate

Protocol (decided before looking at eval numbers):
  1. For each model, choose the cutoff on the VALIDATION set: the strictest cutoff
     that still finds TARGET_RECALL of crack pixels, so models are compared at
     (about) the same recall.
  2. Apply that cutoff to the EVAL set. Report per sample: AP, precision, recall,
     stroke false alarms, impasto false alarms.
  3. Paired comparison, AND-fusion vs every other model, matched by sample id:
     mean difference with a 95% bootstrap interval (the evidence) and a Wilcoxon
     p-value (with n=30, its floor is 1.9e-9: it only says 'every sample agreed').
  4. Stress test: shift the height map 0-4 px against the photo, both directions.

Models: the four classical baselines, plus each U-Net checkpoint found in
checkpoints/ (unet_3ch.pt = photo only, unet_4ch.pt = photo + height).
The U-Nets were trained on seeds 10000+, so they never see the val or eval sets.

Outputs in results/: per_sample.csv, summary.json, pr_curves.png, misregistration.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import stats

from craquelure_fusion.baseline import MODELS as CLASSICAL
from craquelure_fusion.generate import Sample, load_sample
from craquelure_fusion.metrics import false_alarm_rate, tolerant_pr_curve, tolerant_precision_recall

logger = logging.getLogger(__name__)

TARGET_RECALL = 0.90
FUSED = "colour_and_height"
SHIFTS_PX = (0, 1, 2, 3, 4)
METRICS = ("ap", "precision", "recall", "stroke_false_alarm", "impasto_false_alarm")
CHECKPOINTS = {"unet_3ch": Path("checkpoints/unet_3ch.pt"), "unet_4ch": Path("checkpoints/unet_4ch.pt")}

Split = list[tuple[str, Sample]]
Scorer = Callable[[Sample], np.ndarray]
Models = dict[str, Scorer]
Scores = dict[str, list[np.ndarray]]


def default_models(checkpoints: dict[str, Path] = CHECKPOINTS) -> Models:
    """Classical baselines, plus every U-Net whose checkpoint exists."""
    from craquelure_fusion.learned import load_scorer  # imports torch: only when needed

    models: Models = dict(CLASSICAL)
    for name, path in checkpoints.items():
        if path.exists():
            models[name] = load_scorer(path)
        else:
            logger.warning("no checkpoint at %s; %s not evaluated", path, name)
    return models


def load_split(split_dir: Path) -> Split:
    dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())
    return [(d.name, load_sample(d)) for d in dirs]


def score_all(split: Split, models: Models) -> Scores:
    """Every model's score map for every sample, computed once and reused."""
    return {name: [f(s) for _, s in split] for name, f in models.items()}


def choose_thresholds(val: Split, models: Models, val_scores: Scores | None = None) -> dict[str, float]:
    val_scores = val_scores or score_all(val, models)
    truths = [s.crack_mask for _, s in val]
    return {name: tolerant_pr_curve(val_scores[name], truths).threshold_at_recall(TARGET_RECALL)
            for name in models}


def per_sample_rows(eval_set: Split, models: Models, thresholds: dict[str, float],
                    eval_scores: Scores | None = None) -> list[dict]:
    eval_scores = eval_scores or score_all(eval_set, models)
    rows = []
    for i, (sample_id, s) in enumerate(eval_set):
        for name in models:
            score = eval_scores[name][i]
            pred = score >= thresholds[name]
            precision, recall = tolerant_precision_recall(pred, s.crack_mask)
            rows.append({
                "sample": sample_id,
                "model": name,
                "ap": tolerant_pr_curve([score], [s.crack_mask]).average_precision,
                "precision": precision,
                "recall": recall,
                "stroke_false_alarm": false_alarm_rate(pred, s.crack_mask, s.stroke_mask),
                "impasto_false_alarm": false_alarm_rate(pred, s.crack_mask, s.impasto_mask),
            })
    return rows


def means(rows: list[dict], models: Models) -> dict[str, dict[str, float]]:
    """Per model, the mean of each metric over samples (NaN = undefined for that sample, skipped)."""
    return {m: {k: float(np.nanmean([r[k] for r in rows if r["model"] == m])) for k in METRICS}
            for m in models}


def paired_comparison(rows: list[dict], a: str, b: str, metric: str) -> dict:
    """b minus a, matched by sample id. Same samples, so sample-to-sample noise cancels."""
    va = {r["sample"]: r[metric] for r in rows if r["model"] == a}
    vb = {r["sample"]: r[metric] for r in rows if r["model"] == b}
    diff = np.array([vb[k] - va[k] for k in sorted(va) if not (np.isnan(va[k]) or np.isnan(vb[k]))])
    out = {"mean_difference": float(diff.mean()), "n": int(diff.size)}
    if np.allclose(diff, diff[0]):  # no spread: interval and test say nothing
        return {**out, "ci95": [float(diff[0])] * 2, "wilcoxon_p": None}
    ci = stats.bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=5000,
                         method="percentile", random_state=0).confidence_interval
    return {**out, "ci95": [float(ci.low), float(ci.high)], "wilcoxon_p": float(stats.wilcoxon(diff).pvalue)}


def misregistration(eval_set: Split, models: Models) -> dict[str, dict[str, float]]:
    """Pooled AP of every model when the height map is shifted by k px (photo stays put)."""
    truths = [s.crack_mask for _, s in eval_set]
    out: dict[str, dict[str, float]] = {}
    for axis_name, axis in (("horizontal", 1), ("vertical", 0)):
        for k in SHIFTS_PX:
            shifted = [replace(s, height=np.roll(s.height, k, axis=axis)) for _, s in eval_set]
            for name, f in models.items():
                ap = tolerant_pr_curve([f(s) for s in shifted], truths).average_precision
                out.setdefault(f"{name}/{axis_name}", {})[str(k)] = ap
    return out


def pooled_ap(eval_set: Split, models: Models, eval_scores: Scores) -> dict[str, float]:
    truths = [s.crack_mask for _, s in eval_set]
    return {name: tolerant_pr_curve(eval_scores[name], truths).average_precision for name in models}


def save_plots(eval_set: Split, models: Models, eval_scores: Scores, mis: dict,
               out_dir: Path, eval_dir_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truths = [s.crack_mask for _, s in eval_set]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in models:
        c = tolerant_pr_curve(eval_scores[name], truths)
        ax.plot(c.recall, c.precision, label=f"{name} (pooled AP {c.average_precision:.2f})")
    ax.axvline(TARGET_RECALL, color="grey", ls=":", lw=1)
    ax.set(xlabel="recall (cracks found)", ylabel="precision (predictions that are cracks)",
           xlim=(0, 1.02), ylim=(0, 1.02), title=f"{eval_dir_name}: all samples pooled, tolerance 2 px")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for name in models:
        for axis_name, style in (("horizontal", "-"), ("vertical", "--")):
            series = mis[f"{name}/{axis_name}"]
            ax.plot([int(k) for k in series], list(series.values()), style, marker="o", ms=3,
                    label=f"{name}, {axis_name}" if axis_name == "horizontal" or name == FUSED else None)
    ax.set(xlabel="height map shifted by (px)", ylabel="pooled AP", ylim=(0, 1.02),
           title="How alignment limits fusion (dashed = vertical shift)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "misregistration.png", dpi=130)
    plt.close(fig)


def run(val_dir: Path, eval_dir: Path, out_dir: Path, models: Models | None = None) -> dict:
    models = models or default_models()
    val, eval_set = load_split(val_dir), load_split(eval_dir)
    thresholds = choose_thresholds(val, models)
    eval_scores = score_all(eval_set, models)
    rows = per_sample_rows(eval_set, models, thresholds, eval_scores)
    mis = misregistration(eval_set, models)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    summary = {
        "protocol": {"val": str(val_dir), "eval": str(eval_dir), "target_recall": TARGET_RECALL,
                     "tolerance_px": 2, "n_eval_samples": len(eval_set), "models": list(models),
                     "ap_note": "means = mean of per-sample AP; pooled_ap = all samples in one curve"},
        "thresholds_from_val": thresholds,
        "means": means(rows, models),
        "pooled_ap": pooled_ap(eval_set, models, eval_scores),
        "fused_vs": {other: {k: paired_comparison(rows, other, FUSED, k)
                             for k in ("ap", "stroke_false_alarm", "impasto_false_alarm")}
                     for other in models if other != FUSED},
        "misregistration_pooled_ap": mis,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_plots(eval_set, models, eval_scores, mis, out_dir, eval_dir.name)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the experiment on the fixed sets.")
    parser.add_argument("--val", type=Path, default=Path("data/val"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    s = run(args.val, args.eval, args.out)
    print(f"cutoffs chosen on {args.val} at recall {TARGET_RECALL}; scored on {args.eval}\n")
    print(f"{'model':<20}{'AP':>7}{'prec':>7}{'recall':>8}{'stroke FA':>11}{'impasto FA':>12}")
    for m, v in s["means"].items():
        print(f"{m:<20}{v['ap']:>7.3f}{v['precision']:>7.3f}{v['recall']:>8.3f}"
              f"{v['stroke_false_alarm']:>11.3f}{v['impasto_false_alarm']:>12.3f}")
    print(f"\n{FUSED} minus each other model (mean, 95% bootstrap CI):")
    for other, comp in s["fused_vs"].items():
        for k, c in comp.items():
            print(f"  vs {other:<17} {k:<20} {c['mean_difference']:+.3f}  "
                  f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    print("\nAND-fusion pooled AP vs height-map shift:")
    for axis_name in ("horizontal", "vertical"):
        series = s["misregistration_pooled_ap"][f"{FUSED}/{axis_name}"]
        print(f"  {axis_name:<11}", ", ".join(f"{k}px {v:.2f}" for k, v in series.items()))
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
