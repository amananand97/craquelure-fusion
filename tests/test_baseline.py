"""Each model's score map: right shape and range, and fooled by the right confuser."""

from craquelure_fusion.baseline import MODELS, score_colour, score_either, score_fused, score_height
from craquelure_fusion.generate import SampleConfig, generate_sample

S = generate_sample(SampleConfig(size=128, seed=5, n_impasto=1))
STROKE = S.stroke_mask & ~(S.crack_mask | S.impasto_mask)
IMPASTO = S.impasto_mask & ~(S.crack_mask | S.stroke_mask)
CRACK = S.crack_mask & ~(S.stroke_mask | S.impasto_mask)


def test_four_models_return_maps_in_unit_range() -> None:
    assert set(MODELS) == {"colour_only", "height_only", "colour_or_height", "colour_and_height"}
    for name, f in MODELS.items():
        score = f(S)
        assert score.shape == (128, 128), name
        assert 0.0 <= score.min() and score.max() <= 1.0 + 1e-6, name


def test_each_single_sensor_is_fooled_by_its_confuser() -> None:
    assert score_colour(S)[STROKE].mean() > 0.5 * score_colour(S)[CRACK].mean()
    assert score_height(S)[IMPASTO].mean() > 0.5 * score_height(S)[CRACK].mean()


def test_or_fusion_is_fooled_by_both() -> None:
    either = score_either(S)
    assert either[STROKE].mean() > 0.5 * either[CRACK].mean()
    assert either[IMPASTO].mean() > 0.5 * either[CRACK].mean()


def test_and_fusion_is_fooled_by_neither() -> None:
    fused = score_fused(S)
    assert fused[STROKE].mean() < 0.3 * fused[CRACK].mean()
    assert fused[IMPASTO].mean() < 0.3 * fused[CRACK].mean()
