"""Tolerant scores: a line one or two pixels off still counts; a far one does not."""

import numpy as np

from craquelure_fusion.metrics import stroke_false_alarm_rate, tolerant_precision_recall


def _line(col: int) -> np.ndarray:
    m = np.zeros((20, 20), dtype=bool)
    m[:, col] = True
    return m


def test_perfect_prediction_scores_one() -> None:
    p, r = tolerant_precision_recall(_line(5), _line(5))
    assert (p, r) == (1.0, 1.0)


def test_one_pixel_off_still_counts() -> None:
    p, r = tolerant_precision_recall(_line(6), _line(5), tol_px=2)
    assert (p, r) == (1.0, 1.0)


def test_far_off_counts_as_miss() -> None:
    p, r = tolerant_precision_recall(_line(15), _line(5), tol_px=2)
    assert (p, r) == (0.0, 0.0)


def test_stroke_false_alarm_counts_only_stroke_pixels() -> None:
    crack, stroke = _line(3), _line(10)
    pred = _line(10)  # predicts the stroke, not the crack
    assert stroke_false_alarm_rate(pred, crack, stroke) == 1.0
    assert stroke_false_alarm_rate(_line(3), crack, stroke) == 0.0
