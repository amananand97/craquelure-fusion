"""Each model's score map: right shape and range, and fooled by the right confuser.

The mechanism is pinned on a sample with noise, misalignment and impasto shadow switched
off. The last test pins why AND leaks on the real data: the shadow makes impasto a little
dark, so it is no longer 'sunken but not dark'.
"""

from dataclasses import replace

from craquelure_fusion.baseline import MODELS, score_colour, score_either, score_fused, score_height
from craquelure_fusion.generate import SampleConfig, generate_sample

CLEAN = SampleConfig(size=160, height_noise=(0.0, 0.0), rgb_noise=(0.0, 0.0), misregister_px=(0.0, 0.0),
                     impasto_shadow=(0.0, 0.0), n_cracks=(1, 1), n_strokes=(1, 1), n_impasto=(1, 1),
                     width_px=(4.0, 4.0), seed=7)
S = generate_sample(CLEAN)


def _ratio(score, confuser) -> float:
    """Mean score on a confuser, as a fraction of the mean score on the crack."""
    return score[confuser].mean() / score[S.crack_mask].mean()


def test_four_models_return_maps_in_unit_range() -> None:
    assert set(MODELS) == {"colour_only", "height_only", "colour_or_height", "colour_and_height"}
    for name, f in MODELS.items():
        score = f(S)
        assert score.shape == (160, 160), name
        assert 0.0 <= score.min() and score.max() <= 1.0 + 1e-6, name


def test_each_single_sensor_is_fooled_by_its_confuser() -> None:
    assert _ratio(score_colour(S), S.stroke_mask) > 0.5
    assert _ratio(score_height(S), S.impasto_mask) > 0.5


def test_or_fusion_is_fooled_by_both() -> None:
    either = score_either(S)
    assert _ratio(either, S.stroke_mask) > 0.5
    assert _ratio(either, S.impasto_mask) > 0.5


def test_and_fusion_is_fooled_by_neither() -> None:
    fused = score_fused(S)
    assert _ratio(fused, S.stroke_mask) < 0.5
    assert _ratio(fused, S.impasto_mask) < 0.5


def test_impasto_shadow_is_what_makes_and_fusion_leak() -> None:
    shadowed = generate_sample(replace(CLEAN, impasto_shadow=(0.25, 0.25)))
    fused = score_fused(shadowed)
    assert fused[shadowed.impasto_mask].mean() > 0.8 * fused[shadowed.crack_mask].mean()
