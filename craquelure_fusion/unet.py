"""Small U-Net: pixels in, a crack probability per pixel out.

Two versions, chosen by how many input channels it gets:
  3 channels  photo only              (the learned version of 'colour only')
  4 channels  photo + height map      (the learned version of 'colour AND height')

Why U-Net: it sees context at several scales (a thin line only makes sense next to
its surroundings) and gives back a map the same size as the input, so the same
pixel-level metrics used for the classical models apply unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from craquelure_fusion.generate import Sample

HEIGHT_SCALE = 16.0  # height units -> roughly [-2, 2] so it is on the same footing as the photo


class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Encoder (down x depth) -> bottleneck -> decoder (up x depth) with skip connections.

    Input height and width must be divisible by 2**depth (the pooling halves them).
    Output: one logit per pixel, shape (N, 1, H, W). Apply sigmoid for a probability.
    """

    def __init__(self, in_channels: int = 3, base: int = 8, depth: int = 3) -> None:
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([DoubleConv(in_channels, widths[0])] +
                                 [DoubleConv(widths[i - 1], widths[i]) for i in range(1, depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([DoubleConv(2 * widths[i], widths[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            if i > 0:
                x = self.pool(x)
            x = block(x)
            if i < self.depth:
                skips.append(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def make_input(sample: Sample, use_height: bool) -> np.ndarray:
    """(C, H, W) float32. Channels: R, G, B, then height/HEIGHT_SCALE if use_height."""
    chans = [sample.rgb.transpose(2, 0, 1)]
    if use_height:
        chans.append((sample.height / HEIGHT_SCALE)[None])
    return np.concatenate(chans, axis=0).astype(np.float32)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy plus soft Dice.

    BCE alone is dominated by background (cracks are about 1% of pixels), so a model can
    score well by predicting 'no crack' everywhere. Dice works on the overlap, so it keeps
    the rare class in view.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = (2 * inter + smooth) / (prob.sum() + target.sum() + smooth)
    return bce + (1 - dice)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_scores(model: UNet, sample: Sample, use_height: bool) -> np.ndarray:
    """Score map in [0, 1], same size as the photo. Model is put in eval mode."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(make_input(sample, use_height))[None].to(device)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()
