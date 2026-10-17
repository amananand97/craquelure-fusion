"""The experiment as six named steps, so it can be run, watched, and broken on purpose.

Each step has: plain-language lines saying what it does, where its code lives, a
pass condition, and a check. A failed check or a crash stops the run there; later
steps are skipped, because they would be built on something broken.

Faults are for demonstrations. They are applied IN MEMORY (or on a temporary copy):
the committed data is never touched.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from craquelure_fusion.baseline import MODELS, score_fused
from craquelure_fusion.eval_set import verify_eval_set
from craquelure_fusion.evaluate import (
    FUSED, TARGET_RECALL, Split, choose_thresholds, load_split, means, paired_comparison, per_sample_rows,
)
from craquelure_fusion.metrics import tolerant_pr_curve

FAULTS = {
    "tampered_file": "A data file was changed after it was built",
    "height_sensor_off": "The height sensor returned a flat map",
    "model_bug": "Bug: the colour model returns a map of the wrong size",
    "misaligned_6px": "Photo and height map are misaligned by 6 px",
}

FALSE_ALARM_LIMIT = 0.05
SHIFT_BUDGET_PX = 2  # AND-fusion must hold AP >= 0.9 up to this misalignment


class CheckFailed(Exception):
    """A step ran, but its result is not good enough to build on."""


@dataclass
class Context:
    data_dir: Path
    fault: str | None = None
    val: Split = field(default_factory=list)
    eval_set: Split = field(default_factory=list)
    val_scores: dict = field(default_factory=dict)
    eval_scores: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    means: dict = field(default_factory=dict)
    comparisons: dict = field(default_factory=dict)
    shifts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    what: tuple[str, ...]
    code: str
    passes_if: str
    run: Callable[[Context], str]


@dataclass
class StepResult:
    key: str
    title: str
    status: str = "pending"  # pending | running | passed | failed | skipped
    summary: str = ""
    error: str = ""
    where: list[str] = field(default_factory=list)  # file:line in function, outermost first
    seconds: float = 0.0


# --- the six steps -------------------------------------------------------------

def _check_data(ctx: Context) -> str:
    sets = [ctx.data_dir / "val_v2", ctx.data_dir / "eval_v2"]
    with tempfile.TemporaryDirectory() as tmp:
        if ctx.fault == "tampered_file":  # tamper with a COPY, never the real data
            copy = Path(tmp) / "eval_v2"
            shutil.copytree(sets[1], copy)
            (copy / "007" / "mask.png").write_bytes(b"not a png any more")
            sets[1] = copy
        problems = [p for d in sets for p in verify_eval_set(d)]
    if problems:
        kind, path = problems[0].split(" ", 1)
        short = "/".join(Path(path).parts[-3:])  # e.g. eval_v2/007/mask.png
        raise CheckFailed(f"{len(problems)} file(s) do not match their fingerprint: {kind} {short}")
    return "60 samples, every file matches its fingerprint"


def _load(ctx: Context) -> str:
    ctx.val = load_split(ctx.data_dir / "val_v2")
    ctx.eval_set = load_split(ctx.data_dir / "eval_v2")
    if ctx.fault == "height_sensor_off":
        ctx.val = [(i, replace(s, height=np.zeros_like(s.height))) for i, s in ctx.val]
        ctx.eval_set = [(i, replace(s, height=np.zeros_like(s.height))) for i, s in ctx.eval_set]
    if ctx.fault == "misaligned_6px":
        ctx.val = [(i, replace(s, height=np.roll(s.height, 6, axis=1))) for i, s in ctx.val]
        ctx.eval_set = [(i, replace(s, height=np.roll(s.height, 6, axis=1))) for i, s in ctx.eval_set]

    for split_name, split in (("val", ctx.val), ("eval", ctx.eval_set)):
        for sample_id, s in split:
            shapes = {s.rgb.shape[:2], s.height.shape, s.crack_mask.shape, s.stroke_mask.shape, s.impasto_mask.shape}
            if len(shapes) > 1:
                raise CheckFailed(f"{split_name} sample {sample_id}: photo, height and masks differ in size {shapes}")
            if s.height.std() < 1e-6:
                raise CheckFailed(f"{split_name} sample {sample_id}: the height map is completely flat "
                                  "(height sensor missing or off)")
            if s.rgb.std() < 1e-6:
                raise CheckFailed(f"{split_name} sample {sample_id}: the photo is blank")
            if not s.crack_mask.any():
                raise CheckFailed(f"{split_name} sample {sample_id}: no crack in the answer mask")
    return f"{len(ctx.val)} validation + {len(ctx.eval_set)} evaluation samples loaded and sane"


def _score(ctx: Context) -> str:
    models = dict(MODELS)
    if ctx.fault == "model_bug":  # a realistic bug: an off-by-a-few crop inside a model
        colour = models["colour_only"]
        models["colour_only"] = lambda s: colour(s)[:-6, :]
    for split, store in ((ctx.val, ctx.val_scores), (ctx.eval_set, ctx.eval_scores)):
        for name, f in models.items():
            store[name] = [f(s) for _, s in split]
            for (sample_id, s), score in zip(split, store[name]):
                if score.shape != s.crack_mask.shape:
                    raise CheckFailed(f"{name} returned a {score.shape} map for a {s.crack_mask.shape} "
                                      f"sample ({sample_id}); scores no longer line up with pixels")
                if not (score.min() >= 0 and score.max() <= 1 + 1e-6):
                    raise CheckFailed(f"{name} scores leave the 0-1 range on sample {sample_id}")
    return f"{len(models)} models scored every pixel of {len(ctx.val) + len(ctx.eval_set)} samples"


def _cutoffs(ctx: Context) -> str:
    try:
        ctx.thresholds = choose_thresholds(ctx.val, ctx.val_scores)
    except ValueError as e:
        raise CheckFailed(str(e)) from e
    return "cutoffs at 90% recall: " + ", ".join(f"{k} {v:.2f}" for k, v in ctx.thresholds.items())


def _measure(ctx: Context) -> str:
    ctx.rows = per_sample_rows(ctx.eval_set, ctx.thresholds, ctx.eval_scores)
    ctx.means = means(ctx.rows)
    ctx.comparisons = {other: {k: paired_comparison(ctx.rows, other, FUSED, k)
                               for k in ("ap", "stroke_false_alarm", "impasto_false_alarm")}
                       for other in MODELS if other != FUSED}
    fused = ctx.means[FUSED]
    for other, comp in ctx.comparisons.items():
        if comp["ap"]["ci95"][0] <= 0:
            raise CheckFailed(f"AND-fusion is not clearly better than {other}: AP difference "
                              f"{comp['ap']['mean_difference']:+.2f}, interval includes 0")
    for confuser in ("stroke", "impasto"):
        fa = fused[f"{confuser}_false_alarm"]
        if fa > FALSE_ALARM_LIMIT:
            raise CheckFailed(f"AND-fusion calls {fa:.0%} of {confuser} pixels crack "
                              f"(limit {FALSE_ALARM_LIMIT:.0%}); the two sensors no longer agree on where things are")
    return (f"AND-fusion AP {fused['ap']:.2f}, false alarms: strokes {fused['stroke_false_alarm']:.0%}, "
            f"impasto {fused['impasto_false_alarm']:.0%}")


def _stress(ctx: Context) -> str:
    truths = [s.crack_mask for _, s in ctx.eval_set]
    for k in range(0, 5):
        shifted = [replace(s, height=np.roll(s.height, k, axis=1)) for _, s in ctx.eval_set]
        ctx.shifts[k] = tolerant_pr_curve([score_fused(s) for s in shifted], truths).average_precision
    if ctx.shifts[SHIFT_BUDGET_PX] < 0.9:
        raise CheckFailed(f"AND-fusion AP falls to {ctx.shifts[SHIFT_BUDGET_PX]:.2f} at {SHIFT_BUDGET_PX} px shift")
    return "AND-fusion AP by shift: " + ", ".join(f"{k}px {v:.2f}" for k, v in ctx.shifts.items())


STEPS: tuple[Step, ...] = (
    Step("check", "1 · Check the fixed data",
         ("Two fixed sets: 30 validation samples choose settings, 30 evaluation samples report results.",
          "When they were built, every file got a fingerprint (sha256).",
          "Recompute every fingerprint and compare. One changed byte is enough to fail."),
         "eval_set.py · verify_eval_set", "every file matches its fingerprint", _check_data),
    Step("load", "2 · Load and sanity-check",
         ("Read each sample from disk: photo, height map, crack, stroke and impasto masks.",
          "Check the sizes agree and that each sensor actually measured something.",
          "Bad input is stopped here, before any model sees it."),
         "generate.py · load_sample", "same size everywhere; photo and height both vary; every sample has a crack",
         _load),
    Step("score", "3 · Score every pixel",
         ("One detector (black top-hat): 'thin, and darker or lower than its surroundings'.",
          "Four models: colour only, height only, OR (either sensor), AND (both sensors).",
          "Each model gives every pixel a score from 0 to 1."),
         "baseline.py · MODELS", "every score map has the sample's size and stays within 0-1", _score),
    Step("cutoffs", "4 · Choose cutoffs (validation only)",
         ("A score becomes yes/no at a cutoff.",
          f"For each model, take the strictest cutoff that still finds {TARGET_RECALL:.0%} of cracks.",
          "Chosen on validation samples only, so the evaluation set stays unseen."),
         "evaluate.py · choose_thresholds", f"every model reaches {TARGET_RECALL:.0%} recall on validation",
         _cutoffs),
    Step("measure", "5 · Measure on unseen samples",
         ("Apply each cutoff to the 30 evaluation samples.",
          "Measure AP, cracks found, strokes called crack, impasto called crack.",
          "Compare AND with every other model sample by sample, with 95% intervals."),
         "evaluate.py · per_sample_rows, paired_comparison",
         f"AND beats every model on AP (interval above 0) with under {FALSE_ALARM_LIMIT:.0%} false alarms",
         _measure),
    Step("stress", "6 · Stress test: alignment",
         ("Real scanners never line the photo and height map up perfectly.",
          "Shift the height map 0-4 px against the photo and measure AND-fusion again.",
          "Shows how much alignment error the method can take."),
         "pipeline.py · _stress", f"AND-fusion keeps AP at or above 0.9 up to {SHIFT_BUDGET_PX} px", _stress),
)


def _where(exc: BaseException) -> list[str]:
    """Our own code frames in the traceback, outermost first: the path the failure took."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "craquelure_fusion" in f.filename]
    return [f"{Path(f.filename).name}:{f.lineno} in {f.name}()  ->  {(f.line or '').strip()}" for f in frames]


def run_pipeline(
    data_dir: Path, fault: str | None = None,
    on_update: Callable[[list[StepResult]], None] | None = None,
) -> tuple[list[StepResult], Context]:
    """Run every step in order; stop at the first failure and mark the rest skipped."""
    if fault is not None and fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {sorted(FAULTS)}")
    ctx = Context(data_dir=data_dir, fault=fault)
    results = [StepResult(s.key, s.title) for s in STEPS]
    notify = on_update or (lambda _: None)
    notify(results)

    broken = False
    for step, result in zip(STEPS, results):
        if broken:
            result.status = "skipped"
            continue
        result.status = "running"
        notify(results)
        start = time.perf_counter()
        try:
            result.summary = step.run(ctx)
            result.status = "passed"
        except CheckFailed as e:  # the step ran, but its pass condition failed
            result.status, result.error = "failed", str(e)
            result.where = _where(e)
            broken = True
        except Exception as e:  # a crash: report where in our code it happened
            result.status, result.error = "failed", f"{type(e).__name__}: {e}"
            result.where = _where(e)
            broken = True
        result.seconds = time.perf_counter() - start
        notify(results)
    return results, ctx
