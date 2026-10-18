"""v3: harder, more realistic samples. A separate generator, so v1/v2 never change.

What v2 got too clean, and what v3 does instead:
  crack     v2: flat bottom, fixed darkness   v3: V-shaped (deepest at the centre), darkness varies
  stroke    v2: perfectly flat                v3: slightly raised (paint has thickness)
  impasto   v2: as light as the background    v3: raised rims either side, and a little shadow
  darkness  v2: crack == stroke exactly       v3: both drawn from the same range (overlap, not copy)
  sensors   v2: almost no noise, aligned      v3: height + colour noise, height shifted up to 1.5 px
  lines     v2: one of each, 3 px             v3: 1-2 of each, 2-4 px wide

Ground truth stays in PHOTO coordinates: the masks mark where things are in the
image; the height map is the sensor that is slightly misaligned, as in real scans.
Every random draw comes from one generator seeded by cfg.seed: same seed, same sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from craquelure_fusion.generate import KEEP_APART_PX, Sample, random_path_distance

logger = logging.getLogger(__name__)

Range = tuple[float, float]
RIM_PX = 2.0  # width of the raised paint ridge on each side of an impasto valley


@dataclass(frozen=True)
class HardConfig:
    """Every knob is a (low, high) range; each sample draws its own value."""

    size: int = 256
    weave_period_px: float = 6.0
    weave_height: float = 1.0
    n_cracks: tuple[int, int] = (1, 2)
    n_strokes: tuple[int, int] = (1, 2)
    n_impasto: tuple[int, int] = (1, 2)
    width_px: Range = (2.0, 4.0)
    crack_darkness: Range = (0.35, 0.65)
    stroke_darkness: Range = (0.35, 0.65)
    crack_depth: Range = (5.0, 12.0)      # at the centreline; V-shaped across the crack
    impasto_depth: Range = (5.0, 12.0)    # flat-bottomed valley
    impasto_rim: Range = (1.0, 3.0)       # height of the ridges either side
    impasto_shadow: Range = (0.0, 0.25)   # valleys read slightly darker in the photo
    stroke_raise: Range = (0.3, 1.5)      # paint sits on top: a stroke is a little higher
    height_noise: Range = (0.3, 1.2)      # std, height units
    rgb_noise: Range = (0.005, 0.02)      # std, relative brightness
    misregister_px: Range = (0.0, 1.5)    # height map shifted against the photo
    seed: int = 0


def _draw(rng: np.random.Generator, r: Range) -> float:
    return float(rng.uniform(r[0], r[1]))


def _line_avoiding(taken: np.ndarray, width: float, size: int, rng: np.random.Generator,
                   seed: int, tries: int = 300) -> np.ndarray:
    """Distance-to-centreline map of a new line that keeps KEEP_APART_PX from every
    line already drawn (incl. impasto rims). Overlaps would make a pixel two things."""
    near_taken = ndi.distance_transform_edt(~taken) <= KEEP_APART_PX + RIM_PX if taken.any() else taken
    for _ in range(tries):
        dist = random_path_distance(size, rng)
        if not ((dist <= width / 2 + RIM_PX) & near_taken).any():
            return dist
    raise ValueError(f"seed {seed}: no free room for another line after {tries} tries")


def generate_hard_sample(cfg: HardConfig) -> Sample:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.size

    # canvas: the weave is real geometry, seen by both sensors
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    k = 2 * np.pi / cfg.weave_period_px
    weave = np.sin(k * x) * np.sin(k * y)
    height = cfg.weave_height * weave
    ground = np.array([0.93, 0.89, 0.80], dtype=np.float32)
    rgb = np.clip(ground[None, None, :] * (1.0 + 0.03 * weave)[..., None], 0, 1)

    crack = np.zeros((n, n), bool)
    stroke = np.zeros((n, n), bool)
    impasto = np.zeros((n, n), bool)
    taken = np.zeros((n, n), bool)

    def new_line() -> tuple[np.ndarray, float]:
        width = _draw(rng, cfg.width_px)
        dist = _line_avoiding(taken, width, n, rng, cfg.seed)
        return dist, width

    for _ in range(int(rng.integers(cfg.n_cracks[0], cfg.n_cracks[1] + 1))):
        dist, w = new_line()
        core = dist <= w / 2
        rgb[core] *= 1 - _draw(rng, cfg.crack_darkness)
        depth = _draw(rng, cfg.crack_depth)
        height[core] -= depth * (1 - dist[core] / (w / 2))  # V: deepest on the centreline
        crack |= core
        taken |= core

    for _ in range(int(rng.integers(cfg.n_strokes[0], cfg.n_strokes[1] + 1))):
        dist, w = new_line()
        core = dist <= w / 2
        rgb[core] *= 1 - _draw(rng, cfg.stroke_darkness)
        height[core] += _draw(rng, cfg.stroke_raise)  # paint on top: slightly raised
        stroke |= core
        taken |= core

    for _ in range(int(rng.integers(cfg.n_impasto[0], cfg.n_impasto[1] + 1))):
        dist, w = new_line()
        core = dist <= w / 2
        rim = (dist > w / 2) & (dist <= w / 2 + RIM_PX)
        height[core] -= _draw(rng, cfg.impasto_depth)
        height[rim] += _draw(rng, cfg.impasto_rim)
        rgb[core] *= 1 - _draw(rng, cfg.impasto_shadow)  # a little shadow in the valley
        impasto |= core
        taken |= core | rim

    # sensors: each adds its own noise; the height scanner is slightly misaligned
    height = height + rng.normal(0, _draw(rng, cfg.height_noise), (n, n))
    rgb = np.clip(rgb * (1 + rng.normal(0, _draw(rng, cfg.rgb_noise), (n, n)))[..., None], 0, 1)
    shift_len, shift_dir = _draw(rng, cfg.misregister_px), rng.uniform(0, 2 * np.pi)
    height = ndi.shift(height, (shift_len * np.sin(shift_dir), shift_len * np.cos(shift_dir)), order=1, mode="nearest")

    logger.info("Hard sample seed=%d: crack %d px, stroke %d px, impasto %d px, shift %.2f px",
                cfg.seed, crack.sum(), stroke.sum(), impasto.sum(), shift_len)
    return Sample(rgb=rgb.astype(np.float32), height=height.astype(np.float32),
                  crack_mask=crack, stroke_mask=stroke, impasto_mask=impasto)
