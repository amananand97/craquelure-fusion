"""Scores for thin lines.

Pixel-exact matching is too strict for 3-px lines (one pixel off = total miss), so all
scores are TOLERANT: a predicted pixel counts as correct if a true crack pixel is within
TOL_PX of it; a true crack pixel counts as found if a prediction is within TOL_PX of it.

Two views of the same idea:
    tolerant_precision_recall  - for one yes/no prediction
    tolerant_pr_curve          - for a score map, at every possible cutoff at once
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

TOL_PX = 2


def _disk(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def tolerant_precision_recall(
    pred: np.ndarray, truth: np.ndarray, tol_px: int = TOL_PX
) -> tuple[float, float]:
    """precision: share of predicted pixels near a true crack.
    recall: share of true crack pixels near a prediction."""
    if not pred.any() or not truth.any():
        return 0.0, 0.0
    near_truth = ndi.distance_transform_edt(~truth) <= tol_px
    near_pred = ndi.distance_transform_edt(~pred) <= tol_px
    return float(near_truth[pred].mean()), float(near_pred[truth].mean())


def false_alarm_rate(
    pred: np.ndarray, crack: np.ndarray, confuser: np.ndarray, tol_px: int = TOL_PX
) -> float:
    """Share of the confuser (stroke or impasto) that is 'found' as crack.

    Measured exactly like recall, with the same tolerance, so the two numbers are
    on one scale: a detector that cannot tell a confuser from a crack scores about
    the same here as its recall. Confuser pixels within tol_px of a real crack are
    left out, because a correct crack prediction there would count against it.
    """
    near_crack = ndi.distance_transform_edt(~crack) <= tol_px
    only = confuser & ~near_crack
    if not only.any():
        return float("nan")  # no confuser in this sample: undefined, not zero
    if not pred.any():
        return 0.0
    near_pred = ndi.distance_transform_edt(~pred) <= tol_px
    return float(near_pred[only].mean())


@dataclass
class PRCurve:
    thresholds: np.ndarray  # descending
    precision: np.ndarray
    recall: np.ndarray

    @property
    def average_precision(self) -> float:
        """Area under the precision-recall curve (step rule, as in scikit-learn)."""
        recall_steps = np.diff(np.concatenate([[0.0], self.recall]))
        return float(np.sum(recall_steps * self.precision))

    def threshold_at_recall(self, target: float) -> float:
        """The strictest cutoff that still finds `target` of the cracks."""
        ok = np.nonzero(self.recall >= target)[0]
        if not ok.size:
            raise ValueError(f"recall {target} is never reached (max {self.recall.max():.3f})")
        return float(self.thresholds[ok[0]])


def tolerant_pr_curve(
    scores: list[np.ndarray], truths: list[np.ndarray], tol_px: int = TOL_PX
) -> PRCurve:
    """Pooled precision-recall curve over several images, tolerant to tol_px.

    The trick that makes every cutoff cheap:
      precision - a pixel scoring >= t is correct if it lies near the truth (fixed per pixel).
      recall    - a crack pixel is found at cutoff t if the HIGHEST score within tol_px of it
                  is >= t. So one maximum filter per image answers recall for every t.
    Every distinct score is a cutoff, so the curve and AP are exact.
    """
    disk = _disk(tol_px)
    pred_scores, pred_ok, best_near_truth = [], [], []
    for score, truth in zip(scores, truths):
        near_truth = ndi.distance_transform_edt(~truth) <= tol_px
        pred_scores.append(score.ravel())
        pred_ok.append(near_truth.ravel())
        best_near_truth.append(ndi.maximum_filter(score, footprint=disk)[truth])

    s = np.concatenate(pred_scores)
    ok = np.concatenate(pred_ok)
    best = np.sort(np.concatenate(best_near_truth))
    order = np.argsort(s)
    s_sorted, ok_cum = s[order], np.cumsum(ok[order][::-1])[::-1]  # ok count at or above each rank

    thresholds = np.unique(s)[::-1]
    first = np.searchsorted(s_sorted, thresholds, side="left")    # first pixel scoring >= t
    n_pred = s.size - first
    n_ok = np.where(first < s.size, ok_cum[np.minimum(first, s.size - 1)], 0)
    precision = np.divide(n_ok, n_pred, out=np.ones_like(thresholds), where=n_pred > 0)
    recall = (best.size - np.searchsorted(best, thresholds, side="left")) / max(best.size, 1)
    return PRCurve(thresholds=thresholds, precision=precision, recall=recall)
