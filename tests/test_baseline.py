"""Each model's score map: right shape and range, and fooled by the right confuser."""

import numpy as np

from craquelure_fusion.baseline import MODELS, score_colour, score_fused, score_height
from craquelure_fusion.generate import SampleConfig, generate_sample

S = generate_sample(SampleConfig(size=128, seed=5, n_impasto=1))
STROKE = S.stroke_mask & ~(S.crack_mask | S.impasto_mask)
IMPASTO = S.impasto_mask & ~(S.crack_mask | S.stroke_mask)
CRACK = S.crack_mask & ~(S.stroke_mask | S.impasto_mask)


def test_scores_are_maps_in_unit_range() -> None:
    for name, f in MODELS.items():
        score = f(S)
        assert score.shape == (128, 128), name
        assert 0.0 <= score.min() and score.max() <= 1.0 + 1e-6, name


def test_colour_is_fooled_by_stroke_height_by_impasto() -> None:
    assert score_colour(S)[STROKE].mean() > 0.5 * score_colour(S)[CRACK].mean()
    assert score_height(S)[IMPASTO].mean() > 0.5 * score_height(S)[CRACK].mean()


def test_fused_is_fooled_by_neither() -> None:
    fused = score_fused(S)
    assert fused[STROKE].mean() < 0.3 * fused[CRACK].mean()
    assert fused[IMPASTO].mean() < 0.3 * fused[CRACK].mean()
