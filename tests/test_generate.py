"""The generator: reproducible, every realism feature does what its comment claims,
and height survives the 16-bit PNG round trip."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from craquelure_fusion.generate import KEEP_APART_PX, SampleConfig, decode_height, encode_height, generate_sample

# Noise and misalignment switched off, so each feature can be measured cleanly.
CLEAN = SampleConfig(size=160, height_noise=(0.0, 0.0), rgb_noise=(0.0, 0.0), misregister_px=(0.0, 0.0),
                     n_cracks=(1, 1), n_strokes=(1, 1), n_impasto=(1, 1), width_px=(4.0, 4.0), seed=7)


def _background(s):
    near_any = ndi.distance_transform_edt(~(s.crack_mask | s.stroke_mask | s.impasto_mask)) <= 4
    return ~near_any


def test_same_seed_same_sample() -> None:
    a, b = generate_sample(SampleConfig(seed=3)), generate_sample(SampleConfig(seed=3))
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.height, b.height)


def test_crack_is_v_shaped_deepest_at_centre() -> None:
    s = generate_sample(CLEAN)
    centre = s.crack_mask & (ndi.distance_transform_edt(s.crack_mask) >= 2)  # inner pixels
    edge = s.crack_mask & ~centre
    assert s.height[centre].mean() < s.height[edge].mean() - 1.0


def test_stroke_is_dark_and_slightly_raised() -> None:
    s = generate_sample(CLEAN)
    bg = _background(s)
    assert s.height[s.stroke_mask].mean() > s.height[bg].mean() + 0.2
    assert s.rgb.mean(-1)[s.stroke_mask].mean() < s.rgb.mean(-1)[bg].mean() - 0.1


def test_impasto_is_sunken_with_raised_rims() -> None:
    s = generate_sample(CLEAN)
    bg = _background(s)
    rim = (ndi.distance_transform_edt(~s.impasto_mask) <= 2) & ~s.impasto_mask
    assert s.height[s.impasto_mask].mean() < s.height[bg].mean() - 3.0
    assert s.height[rim].mean() > s.height[bg].mean() + 0.5


def test_lines_keep_apart() -> None:
    for seed in range(5):
        s = generate_sample(SampleConfig(size=200, seed=seed))
        for a, b in ((s.crack_mask, s.stroke_mask), (s.crack_mask, s.impasto_mask), (s.stroke_mask, s.impasto_mask)):
            assert ndi.distance_transform_edt(~a)[b].min() > KEEP_APART_PX


def test_height_map_is_shifted_against_the_photo() -> None:
    aligned = generate_sample(CLEAN)
    shifted = generate_sample(SampleConfig(**{**CLEAN.__dict__, "misregister_px": (1.5, 1.5)}))
    assert np.array_equal(aligned.crack_mask, shifted.crack_mask)  # truth stays in photo coordinates
    assert not np.allclose(aligned.height, shifted.height)


def test_height_png_round_trip(tmp_path: Path) -> None:
    height = np.linspace(-20, 20, 64 * 64, dtype=np.float32).reshape(64, 64)
    Image.fromarray(encode_height(height)).save(tmp_path / "h.png")
    back = decode_height(np.array(Image.open(tmp_path / "h.png")))
    assert np.abs(back - height).max() < 1e-3
