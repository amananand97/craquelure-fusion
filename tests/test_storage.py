"""A sample saved to disk and loaded back must give the same masks and height."""

from pathlib import Path

import numpy as np

from craquelure_fusion.generate import SampleConfig, generate_sample, load_sample, save_sample


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    original = generate_sample(SampleConfig(size=96, seed=3, n_impasto=1))
    save_sample(original, tmp_path)
    back = load_sample(tmp_path)
    for name in ("crack_mask", "stroke_mask", "impasto_mask"):
        assert np.array_equal(getattr(back, name), getattr(original, name)), name
    assert np.abs(back.height - original.height).max() < 1e-3
    assert np.abs(back.rgb - original.rgb).max() < 1 / 255 + 1e-6  # 8-bit rounding only


def test_v1_folder_without_impasto_file_loads(tmp_path: Path) -> None:
    save_sample(generate_sample(SampleConfig(size=64)), tmp_path)
    (tmp_path / "impasto_mask.png").unlink()  # what a v1 sample looks like
    assert not load_sample(tmp_path).impasto_mask.any()
