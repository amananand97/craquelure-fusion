"""Procedural generator for synthetic craquelure samples, plus saving and loading.

The rule this whole repo rests on:
    the photo (rgb) and the height map are rendered INDEPENDENTLY from the same
    ground-truth geometry (the masks). Height is never computed from the photo.

A crack is dark AND sunken.
A painted stroke is dark and slightly raised  -> fools a colour-only detector.
An impasto valley is sunken with raised rims  -> fools a height-only detector.

Realism, so the task is not trivially separable:
  cracks    V-shaped (deepest at the centre), darkness varies per crack
  strokes   paint has thickness, so a stroke sits a little higher than the canvas
  impasto   raised rims either side, and a little shadow in the valley
  darkness  cracks and strokes draw their darkness from the same range
  sensors   each adds its own noise; the height map is shifted up to 1.5 px

Ground truth stays in PHOTO coordinates: the masks mark where things are in the
image; the height map is the sensor that is slightly misaligned, as in real scans.
Every random draw comes from one generator seeded by cfg.seed: same seed, same sample.

Run from the repo root:
    python -m craquelure_fusion.generate --out samples/sample_000 --seed 0
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

# Fixed physical range for encoding height into a 16-bit PNG.
# Fixed (not per-image min/max) so the same pixel value means the same
# height in every sample we ever generate.
HEIGHT_RANGE: tuple[float, float] = (-32.0, 32.0)

KEEP_APART_PX = 4   # a line never comes this close to another line
RIM_PX = 2.0        # width of the raised paint ridge on each side of an impasto valley
WALK_STEPS = 400    # length of the random walk that traces a line
WALK_TURN_STD = 0.08  # how much the walk turns per step (radians)
MAX_TRIES = 300     # attempts to place a line before giving up on a seed

Range = tuple[float, float]


@dataclass(frozen=True)
class SampleConfig:
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


@dataclass
class Sample:
    rgb: np.ndarray           # (H, W, 3) float32 in [0, 1]
    height: np.ndarray        # (H, W) float32, height units
    crack_mask: np.ndarray    # (H, W) bool -> mask.png (ground truth)
    stroke_mask: np.ndarray   # (H, W) bool -> confuser that fools COLOUR (dark, flat)
    impasto_mask: np.ndarray  # (H, W) bool -> confuser that fools HEIGHT (sunken, not dark)


# --- generating -------------------------------------------------------------------

def random_path_distance(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random walk -> distance (px) from every pixel to the walked centreline.

    Thresholding the distance gives a line of any width; using the distance itself
    gives a profile across the line (e.g. a V-shaped crack, deepest at the centre).
    """
    pos = rng.uniform(0.2 * size, 0.8 * size, size=2)
    angle = rng.uniform(0, 2 * np.pi)
    centreline = np.zeros((size, size), dtype=bool)

    for _ in range(WALK_STEPS):
        angle += rng.normal(0, WALK_TURN_STD)
        pos += np.array([np.cos(angle), np.sin(angle)])
        row, col = int(round(pos[1])), int(round(pos[0]))
        if not (0 <= row < size and 0 <= col < size):
            break
        centreline[row, col] = True

    return ndi.distance_transform_edt(~centreline)


def _draw(rng: np.random.Generator, r: Range) -> float:
    return float(rng.uniform(r[0], r[1]))


def _line_avoiding(taken: np.ndarray, width: float, size: int, rng: np.random.Generator, seed: int) -> np.ndarray:
    """Distance-to-centreline map of a new line that keeps KEEP_APART_PX from every
    line already drawn (incl. impasto rims). Overlaps would make a pixel two things."""
    near_taken = ndi.distance_transform_edt(~taken) <= KEEP_APART_PX + RIM_PX if taken.any() else taken
    for _ in range(MAX_TRIES):
        dist = random_path_distance(size, rng)
        if not ((dist <= width / 2 + RIM_PX) & near_taken).any():
            return dist
    raise ValueError(f"seed {seed}: no free room for another line after {MAX_TRIES} tries")


def generate_sample(cfg: SampleConfig) -> Sample:
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

    logger.info("Sample seed=%d: crack %d px, stroke %d px, impasto %d px, shift %.2f px",
                cfg.seed, crack.sum(), stroke.sum(), impasto.sum(), shift_len)
    return Sample(rgb=rgb.astype(np.float32), height=height.astype(np.float32),
                  crack_mask=crack, stroke_mask=stroke, impasto_mask=impasto)


# --- saving and loading -----------------------------------------------------------

def encode_height(height: np.ndarray) -> np.ndarray:
    """float height -> uint16 using the FIXED HEIGHT_RANGE."""
    lo, hi = HEIGHT_RANGE
    scaled = (np.clip(height, lo, hi) - lo) / (hi - lo)
    return np.round(scaled * 65535).astype(np.uint16)


def decode_height(encoded: np.ndarray) -> np.ndarray:
    """uint16 -> float height. Inverse of encode_height."""
    lo, hi = HEIGHT_RANGE
    return (encoded.astype(np.float32) / 65535) * (hi - lo) + lo


def save_sample(sample: Sample, out_dir: Path) -> None:
    """Write rgb.png (8-bit), height.png (16-bit), mask.png, stroke_mask.png, impasto_mask.png (0/255)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(sample.rgb * 255).astype(np.uint8)).save(out_dir / "rgb.png")
    Image.fromarray(encode_height(sample.height)).save(out_dir / "height.png")
    Image.fromarray(sample.crack_mask.astype(np.uint8) * 255).save(out_dir / "mask.png")
    Image.fromarray(sample.stroke_mask.astype(np.uint8) * 255).save(out_dir / "stroke_mask.png")
    Image.fromarray(sample.impasto_mask.astype(np.uint8) * 255).save(out_dir / "impasto_mask.png")
    logger.info("Saved rgb, height and the three masks to %s", out_dir)


def load_sample(in_dir: Path) -> Sample:
    """Read a sample back from disk. Evaluation always reads from disk, never from memory,
    so what a model sees is exactly what was saved (including 8-bit rounding of rgb)."""
    rgb = np.array(Image.open(in_dir / "rgb.png").convert("RGB"), dtype=np.float32) / 255.0
    height = decode_height(np.array(Image.open(in_dir / "height.png")))
    crack = np.array(Image.open(in_dir / "mask.png")) > 0
    stroke = np.array(Image.open(in_dir / "stroke_mask.png")) > 0
    impasto = np.array(Image.open(in_dir / "impasto_mask.png")) > 0
    return Sample(rgb=rgb, height=height, crack_mask=crack, stroke_mask=stroke, impasto_mask=impasto)


def save_preview(sample: Sample, path: Path) -> None:
    """rgb | height | mask side by side, for eyeballing."""
    import matplotlib

    matplotlib.use("Agg")  # no window; just write the file
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sample.rgb)
    axes[0].set_title("rgb")
    im = axes[1].imshow(sample.height, cmap="viridis")
    axes[1].set_title("height")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(sample.crack_mask, cmap="gray")
    axes[2].set_title("mask (crack only)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved preview to %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate one synthetic craquelure sample.")
    parser.add_argument("--out", type=Path, default=Path("samples/sample_000"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    sample = generate_sample(SampleConfig(seed=args.seed))
    save_sample(sample, args.out)
    save_preview(sample, args.out / "preview.png")


if __name__ == "__main__":
    main()
