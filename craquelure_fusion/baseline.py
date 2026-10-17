"""Classical baseline: no learning. Each model turns a sample into a SCORE map in [0, 1]
(higher = more crack-like). Turning scores into yes/no is the evaluation's job, so
every model is judged at the same recall instead of at a hand-picked cutoff.

Four models, one question each:
    colour   - is it dark and thin?             (fooled by painted strokes)
    height   - is it sunken and thin?           (fooled by impasto valleys)
    OR       - is it dark OR sunken?            (two sensors, but fooled by both)
    AND      - is it dark AND sunken?           (should be fooled by neither)
The OR model is the control that shows having two sensors is not enough:
how you combine them decides whether the confusers drop out.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage as ndi

from craquelure_fusion.generate import Sample

WINDOW_PX = 9  # keeps structures thinner than this; our lines are 3 px wide


def black_tophat(image: np.ndarray, window_px: int = WINDOW_PX) -> np.ndarray:
    """Highlight dark-or-low structures thinner than window_px.

    Closing fills thin dark lines with the brightness around them; subtracting
    the original leaves exactly those lines, bright in the result.
    """
    return ndi.grey_closing(image, size=(window_px, window_px)) - image


def _normalise(score: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] so scores from different sensors can be combined."""
    return score / (score.max() + 1e-8)


def score_colour(sample: Sample) -> np.ndarray:
    return _normalise(black_tophat(sample.rgb.mean(axis=-1)))  # mean of R, G, B = lightness


def score_height(sample: Sample) -> np.ndarray:
    return _normalise(black_tophat(sample.height))


def score_fused(sample: Sample) -> np.ndarray:
    """Geometric mean = AND: high only where BOTH sensors are high.
    A stroke scores ~0 on height, an impasto valley ~0 on colour, so both drop out."""
    return np.sqrt(score_colour(sample) * score_height(sample))


def score_either(sample: Sample) -> np.ndarray:
    """Maximum = OR: high where EITHER sensor is high. Keeps both confusers."""
    return np.maximum(score_colour(sample), score_height(sample))


MODELS: dict[str, Callable[[Sample], np.ndarray]] = {
    "colour_only": score_colour,
    "height_only": score_height,
    "colour_or_height": score_either,
    "colour_and_height": score_fused,
}
