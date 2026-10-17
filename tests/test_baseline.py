"""The classical baseline: right shape, right amount, and the fusion claim."""

import numpy as np

from craquelure_fusion.baseline import PERCENT_TOP, predict_colour_only, predict_fused
from craquelure_fusion.generate import SampleConfig, generate_sample
from craquelure_fusion.metrics import stroke_false_alarm_rate

CFG = SampleConfig(size=128)


def test_both_models_predict_the_same_amount() -> None:
    s = generate_sample(CFG)
    a, b = predict_colour_only(s.rgb), predict_fused(s.rgb, s.height)
    assert a.shape == b.shape == (128, 128)
    assert a.dtype == b.dtype == bool
    expected = PERCENT_TOP / 100 * a.size
    assert abs(a.sum() - expected) <= 0.02 * a.size
    assert abs(b.sum() - expected) <= 0.02 * b.size


def test_fusion_reduces_stroke_false_alarms() -> None:
    # The claim the project tests, on a small fixed set of seeds.
    a_scores, b_scores = [], []
    for seed in range(1000, 1005):
        s = generate_sample(SampleConfig(size=128, seed=seed))
        a_scores.append(stroke_false_alarm_rate(predict_colour_only(s.rgb), s.crack_mask, s.stroke_mask))
        b_scores.append(stroke_false_alarm_rate(predict_fused(s.rgb, s.height), s.crack_mask, s.stroke_mask))
    assert np.mean(b_scores) < np.mean(a_scores)
