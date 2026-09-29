"""Procedural generator for ONE synthetic craquelure sample.

The rule this whole repo rests on:
    rgb and height are rendered INDEPENDENTLY from the same ground-truth
    geometry (the masks). Height is never computed from the rgb image.

A crack is dark AND sunken. A painted stroke is dark but flat.
If both look alike in rgb but differ in height, fusion has something to find.

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


@dataclass(frozen=True)
class SampleConfig:
    """Every knob for one sample. Same config + same seed = same sample."""

    size: int = 256
    weave_period_px: float = 6.0   # canvas thread spacing
    weave_height: float = 1.0      # canvas bump amplitude (height units)
    line_width_px: float = 3.0     # crack AND stroke share this
    line_darkness: float = 0.55    # fraction of brightness removed inside a line
    crack_depth: float = 12.0      # how far the crack sinks (height units)
    seed: int = 0


@dataclass
class Sample:
    rgb: np.ndarray          # (H, W, 3) float32 in [0, 1]
    height: np.ndarray       # (H, W) float32, height units
    crack_mask: np.ndarray   # (H, W) bool  -> mask.png (ground truth)
    stroke_mask: np.ndarray  # (H, W) bool  -> confuser, used for eval later


# ---------------------------------------------------------------------------
# Plumbing: written for you. Read it, don't rewrite it.
# ---------------------------------------------------------------------------

def make_canvas(cfg: SampleConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Blank primed canvas: returns (rgb, height).

    The weave is physical, so BOTH sensors see it. Each gets its own
    independent noise, which is what two real instruments would give you.
    """
    y, x = np.mgrid[0 : cfg.size, 0 : cfg.size].astype(np.float32)
    k = 2 * np.pi / cfg.weave_period_px
    weave = np.sin(k * x) * np.sin(k * y)  # [-1, 1], the world's geometry

    height = cfg.weave_height * weave + rng.normal(0, 0.1 * cfg.weave_height, weave.shape)

    ground = np.array([0.93, 0.89, 0.80], dtype=np.float32)  # warm off-white
    shade = 1.0 + 0.03 * weave + rng.normal(0, 0.01, weave.shape)
    rgb = np.clip(ground[None, None, :] * shade[..., None], 0.0, 1.0)

    return rgb.astype(np.float32), height.astype(np.float32)


def random_path_mask(
    size: int,
    width_px: float,
    rng: np.random.Generator,
    n_steps: int = 400,
    turn_std: float = 0.08,
) -> np.ndarray:
    """A thin wandering line as a boolean mask (random walk, then thickened)."""
    pos = rng.uniform(0.2 * size, 0.8 * size, size=2)
    angle = rng.uniform(0, 2 * np.pi)
    centreline = np.zeros((size, size), dtype=bool)

    for _ in range(n_steps):
        angle += rng.normal(0, turn_std)
        pos += np.array([np.cos(angle), np.sin(angle)])
        row, col = int(round(pos[1])), int(round(pos[0]))
        if not (0 <= row < size and 0 <= col < size):
            break
        centreline[row, col] = True

    # distance from every pixel to the nearest centreline pixel
    dist = ndi.distance_transform_edt(~centreline)
    return dist <= width_px / 2


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
    """Write rgb.png (8-bit), height.png (16-bit), mask.png (0/255)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(sample.rgb * 255).astype(np.uint8)).save(out_dir / "rgb.png")
    Image.fromarray(encode_height(sample.height)).save(out_dir / "height.png")
    Image.fromarray(sample.crack_mask.astype(np.uint8) * 255).save(out_dir / "mask.png")
    logger.info("Saved rgb.png, height.png, mask.png to %s", out_dir)


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


# ---------------------------------------------------------------------------
# YOUR PART: three TODOs. These are the decisions that make the data honest.
# ---------------------------------------------------------------------------

def paint_line(rgb: np.ndarray, mask: np.ndarray, darkness: float) -> np.ndarray:
    """Darken rgb inside `mask` by `darkness` (0 = no change, 1 = black).

    Called for BOTH the crack and the stroke. That shared call is why the two
    look alike in colour: same function, same darkness, same width.

    Must return a NEW array. The input must be unchanged afterwards.
    """
    # TODO(1): copy rgb, multiply the masked pixels by (1 - darkness), return the copy.
    #   rgb has shape (H, W, 3); mask has shape (H, W).
    #   rgb[mask] selects the masked pixels as an (N, 3) array.
    raise NotImplementedError("TODO(1)")


def carve_crack(height: np.ndarray, mask: np.ndarray, depth: float) -> np.ndarray:
    """Sink the surface by `depth` inside `mask`. Flat-bottomed for v1.

    Must return a NEW array. The input must be unchanged afterwards.
    """
    # TODO(2): same pattern as TODO(1), but for a 2-D array, and subtract instead of multiply.
    raise NotImplementedError("TODO(2)")


def generate_sample(cfg: SampleConfig) -> Sample:
    """Build one sample: canvas + one crack + one painted stroke."""
    rng = np.random.default_rng(cfg.seed)

    rgb, height = make_canvas(cfg, rng)
    crack = random_path_mask(cfg.size, cfg.line_width_px, rng)
    stroke = random_path_mask(cfg.size, cfg.line_width_px, rng)

    # Crack: dark AND sunken.
    rgb = paint_line(rgb, crack, cfg.line_darkness)
    height = carve_crack(height, crack, cfg.crack_depth)

    # Stroke: dark, flat.
    # TODO(3): apply the stroke. Which array changes, and which stays alone?
    #   Write the line, then a one-line comment saying WHY.
    raise NotImplementedError("TODO(3)")

    logger.info(
        "Sample seed=%d: crack %d px, stroke %d px, overlap %d px",
        cfg.seed,
        crack.sum(),
        stroke.sum(),
        (crack & stroke).sum(),
    )
    return Sample(rgb=rgb, height=height, crack_mask=crack, stroke_mask=stroke)


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
