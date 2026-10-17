"""A sample saved to disk and loaded back must give the same masks and height."""

from pathlib import Path

import numpy as np

from craquelure_fusion.generate import SampleConfig, generate_sample, load_sample, save_sample


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    original = generate_sample(SampleConfig(size=96, seed=3))
    save_sample(original, tmp_path)
    back = load_sample(tmp_path)
    assert np.array_equal(back.crack_mask, original.crack_mask)
    assert np.array_equal(back.stroke_mask, original.stroke_mask)
    assert np.abs(back.height - original.height).max() < 1e-3
    assert np.abs(back.rgb - original.rgb).max() < 1 / 255 + 1e-6  # 8-bit rounding only
