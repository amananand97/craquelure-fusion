"""Classical baseline: no learning. Two detectors and one fusion rule.

Model A (colour only):   finds dark, thin lines in the photo.
Model B (colour+height): finds dark, thin lines AND sunken, thin valleys in the
                         height map, and keeps only the pixels both agree on.

Both predict the SAME number of pixels (the top PERCENT_TOP of their score),
so the comparison is fair: any difference comes from the information, not from
one model predicting more.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

WINDOW_PX = 9       # structures thinner than this (our lines are 3 px) are kept
PERCENT_TOP = 1.0   # predict the top 1% of pixels as crack


def black_tophat(image: np.ndarray, window_px: int = WINDOW_PX) -> np.ndarray:
    """Highlight dark-or-low structures that are thinner than window_px.

    Closing fills in dark thin lines with the brightness around them;
    subtracting the original leaves exactly those lines, bright in the result.
    """
    closed = ndi.grey_closing(image, size=(window_px, window_px))
    return closed - image


def _normalise(score: np.ndarray) -> np.ndarray:
    """Scale to 0..1 so two scores can be multiplied fairly."""
    return score / (score.max() + 1e-8)


def _top_percent(score: np.ndarray, percent: float = PERCENT_TOP) -> np.ndarray:
    """True for the highest-scoring `percent` of pixels."""
    cutoff = np.percentile(score, 100.0 - percent)
    return score >= cutoff


def predict_colour_only(rgb: np.ndarray) -> np.ndarray:
    """Model A. rgb: (H, W, 3) in [0, 1]. Returns (H, W) bool."""
    darkness = black_tophat(rgb.mean(axis=-1))  # mean over R, G, B = lightness
    return _top_percent(darkness)


def predict_fused(rgb: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Model B. Crack = dark AND sunken, combined as a geometric mean.

    The geometric mean is high only where BOTH scores are high. That is the
    'AND' in one line: a painted stroke is dark but flat, so its height score
    is near zero and the product is near zero.
    """
    dark = _normalise(black_tophat(rgb.mean(axis=-1)))
    sunken = _normalise(black_tophat(height))
    fused = np.sqrt(dark * sunken)
    return _top_percent(fused)
