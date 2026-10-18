"""The U-Net step loads and scores; the measure gate checks the single-sensor rules and reports the rest."""

from pathlib import Path

import pytest

from craquelure_fusion import pipeline
from craquelure_fusion.baseline import MODELS
from craquelure_fusion.pipeline import Context

DATA = Path(__file__).resolve().parents[1] / "data"
UNETS = ("unet_3ch", "unet_4ch")


def test_unet_step_loads_and_scores_when_checkpoints_exist() -> None:
    if not all((DATA.parent / "checkpoints" / f"{n}.pt").exists() for n in UNETS):
        pytest.skip("U-Net checkpoints are not on this machine")
    ctx = Context(data_dir=DATA)
    pipeline._load(ctx)
    assert pipeline._unet(ctx) == "both U-Nets loaded and scored every pixel"
    for name in UNETS:
        assert name in ctx.models
        assert len(ctx.val_scores[name]) == len(ctx.val)
        assert len(ctx.eval_scores[name]) == len(ctx.eval_set)


def _ready_context() -> Context:
    """Loaded data with every classical model scored, as the pipeline would leave it before cutoffs."""
    ctx = Context(data_dir=DATA)
    pipeline._load(ctx)
    ctx.models = dict(MODELS)
    for split, store in ((ctx.val, ctx.val_scores), (ctx.eval_set, ctx.eval_scores)):
        for name, f in ctx.models.items():
            store[name] = [f(s) for _, s in split]
    return ctx


def _add_oracle(ctx: Context, name: str) -> None:
    """A model that returns the answer itself: perfect AP. Used to test the gate, not to claim anything."""
    def oracle(s):
        return s.crack_mask.astype(float)
    ctx.models[name] = oracle
    for split, store in ((ctx.val, ctx.val_scores), (ctx.eval_set, ctx.eval_scores)):
        store[name] = [oracle(s) for _, s in split]


def test_a_learned_model_beating_and_is_reported_not_gated() -> None:
    ctx = _ready_context()
    _add_oracle(ctx, "unet_fake")
    pipeline._cutoffs(ctx)
    logs: list[str] = []
    ctx.say = logs.append
    pipeline._measure(ctx)  # must not raise
    assert "unet_fake" in ctx.comparisons
    assert any("reported, not gated: unet_fake" in line for line in logs)


def test_gate_fails_when_and_does_not_beat_a_single_sensor_rule() -> None:
    ctx = _ready_context()
    _add_oracle(ctx, "colour_only")  # a perfect single-sensor rule must beat AND, and the gate must say so
    pipeline._cutoffs(ctx)
    with pytest.raises(pipeline.CheckFailed, match="colour_only"):
        pipeline._measure(ctx)
