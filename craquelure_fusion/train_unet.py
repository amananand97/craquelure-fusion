"""Train a U-Net on freshly generated samples. Run:
    python -m craquelure_fusion.train_unet --channels photo+height --steps 2000 --out checkpoints/unet_4ch.pt

Training samples are generated on the fly from seeds TRAIN_SEED_START and up. They are
never written to disk, and their seeds never overlap the validation (2000+) or evaluation
(3000+) sets, so the reported numbers stay unseen by the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from craquelure_fusion.generate import SampleConfig, generate_sample
from craquelure_fusion.unet import UNet, dice_bce_loss, make_input, pick_device

logger = logging.getLogger(__name__)

TRAIN_SEED_START = 10000  # above every validation and evaluation seed (see eval_set.SEED_START)


@dataclass(frozen=True)
class TrainConfig:
    use_height: bool = True
    steps: int = 2000
    batch: int = 8
    crop: int = 128          # train on random crops: faster, and more variety per sample
    base: int = 8
    depth: int = 3
    lr: float = 2e-3
    seed: int = 0            # controls crops and weight init, not the training sample seeds


def _crop_batch(samples, rng: np.random.Generator, crop: int):
    xs, ys = [], []
    for s in samples:
        x = make_input(s, use_height=True)
        y = s.crack_mask.astype(np.float32)
        r, c = rng.integers(0, x.shape[1] - crop + 1), rng.integers(0, x.shape[2] - crop + 1)
        xs.append(x[:, r:r + crop, c:c + crop])
        ys.append(y[r:r + crop, c:c + crop])
    return xs, ys


def _write_progress(out: Path, step: int, steps: int, loss: float, seeds_used: int, state: str) -> None:
    """Small file the page reads to show live training progress: <name>.progress.json next to the checkpoint."""
    progress = out.with_suffix(".progress.json")
    progress.parent.mkdir(parents=True, exist_ok=True)  # checkpoints/ may not exist yet
    progress.write_text(json.dumps({"step": step, "steps": steps, "loss": float(loss), "seeds_used": seeds_used,
                                    "state": state, "updated": time.strftime("%H:%M:%S")}))


def train(cfg: TrainConfig, out: Path | None = None) -> list[float]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device()
    in_ch = 4 if cfg.use_height else 3
    model = UNet(in_channels=in_ch, base=cfg.base, depth=cfg.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses: list[float] = []
    next_seed = TRAIN_SEED_START

    for step in range(cfg.steps):
        samples = []
        for _ in range(cfg.batch):
            samples.append(generate_sample(SampleConfig(seed=next_seed)))
            next_seed += 1
        xs, ys = _crop_batch(samples, rng, cfg.crop)
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys))[:, None].to(device)
        if not cfg.use_height:
            x = x[:, :3]

        model.train()
        loss = dice_bce_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == cfg.steps - 1:
            logger.info("step %d/%d  loss %.4f  (device %s)", step, cfg.steps, losses[-1], device)
        if out is not None and (step % 10 == 0 or step == cfg.steps - 1):
            _write_progress(out, step + 1, cfg.steps, losses[-1], next_seed - TRAIN_SEED_START, "training")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "train_seeds": [TRAIN_SEED_START, next_seed]}, out)
        logger.info("saved checkpoint to %s (trained on seeds %d-%d)", out, TRAIN_SEED_START, next_seed - 1)
        _write_progress(out, cfg.steps, cfg.steps, losses[-1], next_seed - TRAIN_SEED_START, "done")
    return losses


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a U-Net on generated samples.")
    p.add_argument("--channels", choices=["photo", "photo+height"], default="photo+height")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # one line per training sample would bury the terminal: keep only the every-50-steps lines
    logging.getLogger("craquelure_fusion.generate").setLevel(logging.WARNING)
    cfg = TrainConfig(use_height=args.channels == "photo+height", steps=args.steps)
    train(cfg, args.out)


if __name__ == "__main__":
    main()
