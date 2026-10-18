"""Turn a saved U-Net checkpoint into a scorer with the same signature as the classical
models (Sample -> score map), so the evaluation and the page treat it the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from craquelure_fusion.generate import Sample
from craquelure_fusion.unet import UNet, pick_device, predict_scores


def load_scorer(checkpoint: Path) -> Callable[[Sample], np.ndarray]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    device = pick_device()
    net = UNet(in_channels=4 if cfg["use_height"] else 3, base=cfg["base"], depth=cfg["depth"])
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    use_height = cfg["use_height"]
    return lambda sample: predict_scores(net, sample, use_height)
