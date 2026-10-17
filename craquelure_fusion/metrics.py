"""Scores for thin lines. Pixel-exact IoU is too strict: a line one pixel off counts as a miss.

So we use TOLERANT precision and recall: a predicted pixel is correct if a true
crack pixel is within TOL_PX of it, and vice versa.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

TOL_PX = 2


def tolerant_precision_recall(
    pred: np.ndarray, truth: np.ndarray, tol_px: int = TOL_PX
) -> tuple[float, float]:
    """precision: share of predicted pixels near a true crack.
    recall: share of true crack pixels near a prediction."""
    if not pred.any() or not truth.any():
        return 0.0, 0.0
    dist_to_truth = ndi.distance_transform_edt(~truth)
    dist_to_pred = ndi.distance_transform_edt(~pred)
    precision = float((dist_to_truth[pred] <= tol_px).mean())
    recall = float((dist_to_pred[truth] <= tol_px).mean())
    return precision, recall


def stroke_false_alarm_rate(pred: np.ndarray, crack: np.ndarray, stroke: np.ndarray) -> float:
    """Share of painted-stroke pixels (that are not also crack) predicted as crack.
    This is the number the whole project is about: the confuser."""
    stroke_only = stroke & ~crack
    if not stroke_only.any():
        return 0.0
    return float(pred[stroke_only].mean())
