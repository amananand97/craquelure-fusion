"""Tolerant scores: a line one or two pixels off still counts; a far one does not."""

import numpy as np

from craquelure_fusion.metrics import false_alarm_rate, tolerant_pr_curve, tolerant_precision_recall


def _line(col: int, size: int = 20) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[:, col] = True
    return m


def test_perfect_prediction_scores_one() -> None:
    assert tolerant_precision_recall(_line(5), _line(5)) == (1.0, 1.0)


def test_one_pixel_off_still_counts() -> None:
    assert tolerant_precision_recall(_line(6), _line(5), tol_px=2) == (1.0, 1.0)


def test_far_off_counts_as_miss() -> None:
    assert tolerant_precision_recall(_line(15), _line(5), tol_px=2) == (0.0, 0.0)


def test_false_alarm_counts_the_confuser_only() -> None:
    crack, stroke = _line(3), _line(12)
    assert false_alarm_rate(_line(12), crack, stroke) == 1.0  # predicted the stroke
    assert false_alarm_rate(_line(3), crack, stroke) == 0.0   # predicted the crack


def test_pr_curve_perfect_score_has_ap_one() -> None:
    truth = _line(5)
    c = tolerant_pr_curve([truth.astype(float)], [truth])
    assert c.average_precision > 0.99


def test_pr_curve_matches_single_threshold_definition() -> None:
    rng = np.random.default_rng(0)
    truth = _line(8, size=40)
    score = rng.random((40, 40)) * 0.5 + truth * 0.4
    c = tolerant_pr_curve([score], [truth])
    i = len(c.thresholds) // 2
    p, r = tolerant_precision_recall(score >= c.thresholds[i], truth)
    assert abs(p - c.precision[i]) < 1e-9 and abs(r - c.recall[i]) < 1e-9


def test_threshold_at_recall_reaches_target() -> None:
    rng = np.random.default_rng(1)
    truth = _line(8, size=40)
    score = rng.random((40, 40)) * 0.5 + truth * 0.4
    c = tolerant_pr_curve([score], [truth])
    t = c.threshold_at_recall(0.9)
    assert tolerant_precision_recall(score >= t, truth)[1] >= 0.9
