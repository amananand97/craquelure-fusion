"""Tests for the sample generator: shapes, saving, and the physics
(crack and stroke alike in colour, different in height)."""

from pathlib import Path

import numpy as np
from PIL import Image

from craquelure_fusion.generate import (
    SampleConfig,
    decode_height,
    encode_height,
    generate_sample,
    make_canvas,
    paint_line,
    random_path_mask,
)

CFG = SampleConfig(size=128, seed=1)


def _lightness(rgb: np.ndarray) -> np.ndarray:
    """Plain mean over channels. Good enough to compare two lines on the same ground."""
    return rgb.mean(axis=-1)


# --- plumbing ---------------------------------------------------------------

def test_canvas_shapes_and_ranges() -> None:
    rgb, height = make_canvas(CFG, np.random.default_rng(0))
    assert rgb.shape == (128, 128, 3)
    assert height.shape == (128, 128)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_path_mask_is_thin_and_nonempty() -> None:
    mask = random_path_mask(128, 3.0, np.random.default_rng(0))
    assert mask.dtype == bool
    assert 0 < mask.sum() < 0.2 * mask.size


def test_height_png_round_trip(tmp_path: Path) -> None:
    height = np.linspace(-20, 20, 64 * 64, dtype=np.float32).reshape(64, 64)
    Image.fromarray(encode_height(height)).save(tmp_path / "h.png")
    back = decode_height(np.array(Image.open(tmp_path / "h.png")))
    assert np.abs(back - height).max() < 1e-3


# --- the three marking decisions ----------------------------------------------

def test_paint_line_does_not_mutate_input() -> None:
    rgb, _ = make_canvas(CFG, np.random.default_rng(0))
    original = rgb.copy()
    mask = random_path_mask(128, 3.0, np.random.default_rng(0))
    out = paint_line(rgb, mask, 0.5)
    assert np.array_equal(rgb, original), "paint_line changed its input"
    assert out[mask].mean() < rgb[mask].mean()


def test_crack_is_sunken() -> None:
    s = generate_sample(CFG)
    background = ~(s.crack_mask | s.stroke_mask)
    assert s.height[s.crack_mask].mean() < s.height[background].mean() - 0.5 * CFG.crack_depth


def test_stroke_is_flat() -> None:
    s = generate_sample(CFG)
    stroke_only = s.stroke_mask & ~s.crack_mask
    background = ~(s.crack_mask | s.stroke_mask)
    diff = abs(s.height[stroke_only].mean() - s.height[background].mean())
    assert diff < 0.5 * CFG.weave_height


def test_crack_and_stroke_alike_in_rgb() -> None:
    s = generate_sample(CFG)
    L = _lightness(s.rgb)
    crack_only = s.crack_mask & ~s.stroke_mask
    stroke_only = s.stroke_mask & ~s.crack_mask
    assert abs(L[crack_only].mean() - L[stroke_only].mean()) < 0.03


def test_crack_and_stroke_differ_in_height() -> None:
    s = generate_sample(CFG)
    crack_only = s.crack_mask & ~s.stroke_mask
    stroke_only = s.stroke_mask & ~s.crack_mask
    assert s.height[crack_only].mean() < s.height[stroke_only].mean() - 0.5 * CFG.crack_depth


def test_same_seed_same_sample() -> None:
    a, b = generate_sample(CFG), generate_sample(CFG)
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.height, b.height)
