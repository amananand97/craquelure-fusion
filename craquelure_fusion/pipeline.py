"""The experiment as nine named steps in four phases, so it can be run, watched, and broken on purpose.

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
import json
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
# The thesis: AND must beat each single-sensor rule and OR. The U-Nets are reported beside it, not gated:
# a learned model beating the hand rule is a finding to show, not a failure of the pipeline.
GATED_RIVALS = ("colour_only", "height_only", "colour_or_height")


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
    models: dict = field(default_factory=dict)  # name -> scorer: classical, plus U-Nets once trained
    thresholds: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    means: dict = field(default_factory=dict)
    comparisons: dict = field(default_factory=dict)
    shifts: dict = field(default_factory=dict)
    say: Callable[[str], None] = field(default=lambda msg: None)  # a step reports what it is doing


class NotReady(Exception):
    """A step that cannot run yet (for example, a model not trained). Shown as 'waiting', not failed."""


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    what: tuple[str, ...]
    code: str
    passes_if: str
    run: Callable[[Context], str]
    group: str


@dataclass
class StepResult:
    key: str
    title: str
    status: str = "pending"  # pending | running | passed | failed | skipped | waiting | held
    summary: str = ""
    error: str = ""
    where: list[str] = field(default_factory=list)  # file:line in function, outermost first
    seconds: float = 0.0
    log: list[str] = field(default_factory=list)  # what the step did, line by line, as it happens


# --- the steps ------------------------------------------------------------------

def _generate(ctx: Context) -> str:
    from craquelure_fusion.generate import SampleConfig, generate_sample

    ctx.say("generating seed 3000 twice")
    a = generate_sample(SampleConfig(seed=3000))
    b = generate_sample(SampleConfig(seed=3000))
    if not (np.array_equal(a.rgb, b.rgb) and np.array_equal(a.height, b.height)):
        raise CheckFailed("the same seed made two different samples: the generator is not reproducible")
    if not a.crack_mask.any():
        raise CheckFailed("the generated sample has no crack in it")
    ctx.say(f"generated twice: identical pixels, {int(a.crack_mask.sum())} crack pixels")
    return (f"seed 3000 makes a sample with {int(a.crack_mask.sum())} crack pixels; "
            "generating it twice gives identical pixels")


def _rebuild(ctx: Context) -> str:
    """Regenerate the first 3 evaluation samples from their seeds and compare pixels.

    Pixels, not file bytes: the same pixels can be compressed to different PNG bytes on
    different machines, so a byte comparison would fail for the wrong reason."""
    from craquelure_fusion.eval_set import SEED_START
    from craquelure_fusion.generate import SampleConfig, decode_height, encode_height, generate_sample, load_sample

    compared = 0
    for i in range(3):
        seed = SEED_START["eval"] + i
        stored = load_sample(ctx.data_dir / "eval" / f"{i:03d}")
        fresh = generate_sample(SampleConfig(seed=seed))
        same = (np.array_equal(np.round(fresh.rgb * 255) / 255, stored.rgb)
                and np.array_equal(decode_height(encode_height(fresh.height)), stored.height)
                and np.array_equal(fresh.crack_mask, stored.crack_mask)
                and np.array_equal(fresh.stroke_mask, stored.stroke_mask)
                and np.array_equal(fresh.impasto_mask, stored.impasto_mask))
        if not same:
            raise CheckFailed(f"eval sample {i:03d} no longer rebuilds to the stored pixels")
        compared += 1
        ctx.say(f"eval sample {i:03d}: rebuilt from seed {seed}, pixels match")
    return f"rebuilt {compared} stored samples from their seeds; pixels identical"


def _unet(ctx: Context) -> str:
    """Score with the trained U-Nets. Waits until both checkpoints exist; reports live training progress."""
    ckpt = ctx.data_dir.parent / "checkpoints"
    ready = True
    for name in ("unet_3ch", "unet_4ch"):
        if (ckpt / f"{name}.pt").exists():
            ctx.say(f"{name}: checkpoint saved")
            continue
        ready = False
        prog = ckpt / f"{name}.progress.json"
        if prog.exists():
            p = json.loads(prog.read_text())
            if p.get("state") == "done":
                ctx.say(f"{name}: training finished, checkpoint not yet written")
            else:
                ctx.say(f"{name}: training step {p['step']}/{p['steps']}, loss {p['loss']:.4f}, "
                        f"{p['seeds_used']} fresh training samples so far (at {p['updated']})")
        else:
            ctx.say(f"{name}: training not started, or started without progress logging")
    if not ready:
        raise NotReady("U-Net still training (live progress is shown above)")
    from craquelure_fusion.learned import load_scorer

    for name in ("unet_3ch", "unet_4ch"):
        scorer = load_scorer(ckpt / f"{name}.pt")
        ctx.models[name] = scorer
        for store, split in ((ctx.val_scores, ctx.val), (ctx.eval_scores, ctx.eval_set)):
            store[name] = [scorer(s) for _, s in split]
            for (sample_id, s), score in zip(split, store[name]):
                if score.shape != s.crack_mask.shape:
                    raise CheckFailed(f"{name} returned a {score.shape} map for a {s.crack_mask.shape} "
                                      f"sample ({sample_id})")
                if not (score.min() >= 0 and score.max() <= 1 + 1e-6):
                    raise CheckFailed(f"{name} scores leave the 0-1 range on sample {sample_id}")
        ctx.say(f"{name}: loaded, scored {len(ctx.val)} validation and {len(ctx.eval_set)} evaluation samples")
    return "both U-Nets loaded and scored every pixel"


def _check_data(ctx: Context) -> str:
    sets = [ctx.data_dir / "val", ctx.data_dir / "eval"]
    with tempfile.TemporaryDirectory() as tmp:
        if ctx.fault == "tampered_file":  # tamper with a COPY, never the real data
            copy = Path(tmp) / "eval"
            shutil.copytree(sets[1], copy)
            (copy / "007" / "mask.png").write_bytes(b"not a png any more")
            sets[1] = copy
        problems = [p for d in sets for p in verify_eval_set(d)]
    for d in sets:
        ctx.say(f"{d.name}: fingerprints checked")
    if problems:
        kind, path = problems[0].split(" ", 1)
        short = "/".join(Path(path).parts[-3:])  # e.g. eval/007/mask.png
        raise CheckFailed(f"{len(problems)} file(s) do not match their fingerprint: {kind} {short}")
    return "60 samples, every file matches its fingerprint"


def _load(ctx: Context) -> str:
    ctx.val = load_split(ctx.data_dir / "val")
    ctx.eval_set = load_split(ctx.data_dir / "eval")
    if ctx.fault == "height_sensor_off":
        ctx.val = [(i, replace(s, height=np.zeros_like(s.height))) for i, s in ctx.val]
        ctx.eval_set = [(i, replace(s, height=np.zeros_like(s.height))) for i, s in ctx.eval_set]
    if ctx.fault == "misaligned_6px":
        ctx.val = [(i, replace(s, height=np.roll(s.height, 6, axis=1))) for i, s in ctx.val]
        ctx.eval_set = [(i, replace(s, height=np.roll(s.height, 6, axis=1))) for i, s in ctx.eval_set]

    ctx.say(f"loaded {len(ctx.val)} validation and {len(ctx.eval_set)} evaluation samples from disk")
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
    ctx.models = dict(models)
    for split, store in ((ctx.val, ctx.val_scores), (ctx.eval_set, ctx.eval_scores)):
        for name, f in models.items():
            store[name] = [f(s) for _, s in split]
            ctx.say(f"{name}: scored {len(split)} samples")
            for (sample_id, s), score in zip(split, store[name]):
                if score.shape != s.crack_mask.shape:
                    raise CheckFailed(f"{name} returned a {score.shape} map for a {s.crack_mask.shape} "
                                      f"sample ({sample_id}); scores no longer line up with pixels")
                if not (score.min() >= 0 and score.max() <= 1 + 1e-6):
                    raise CheckFailed(f"{name} scores leave the 0-1 range on sample {sample_id}")
    return f"{len(models)} models scored every pixel of {len(ctx.val) + len(ctx.eval_set)} samples"


def _cutoffs(ctx: Context) -> str:
    try:
        ctx.thresholds = choose_thresholds(ctx.val, ctx.models, ctx.val_scores)
    except ValueError as e:
        raise CheckFailed(str(e)) from e
    for k, v in ctx.thresholds.items():
        ctx.say(f"{k}: cutoff {v:.2f} keeps {TARGET_RECALL:.0%} of validation cracks")
    return "cutoffs at 90% recall: " + ", ".join(f"{k} {v:.2f}" for k, v in ctx.thresholds.items())


def _measure(ctx: Context) -> str:
    ctx.rows = per_sample_rows(ctx.eval_set, ctx.models, ctx.thresholds, ctx.eval_scores)
    ctx.means = means(ctx.rows, ctx.models)
    ctx.comparisons = {other: {k: paired_comparison(ctx.rows, other, FUSED, k)
                               for k in ("ap", "stroke_false_alarm", "impasto_false_alarm")}
                       for other in ctx.models if other != FUSED}
    ctx.say(f"measured {len(ctx.eval_set)} unseen samples with the cutoffs above")
    for other in ctx.models:
        if other not in GATED_RIVALS and other != FUSED:
            ctx.say(f"reported, not gated: {other} AP {ctx.means[other]['ap']:.3f} "
                    f"(AND {ctx.means[FUSED]['ap']:.3f})")
    fused = ctx.means[FUSED]
    for other in GATED_RIVALS:
        comp = ctx.comparisons[other]
        if comp["ap"]["ci95"][0] <= 0:
            raise CheckFailed(f"AND-fusion is not clearly better than {other}: AP difference "
                              f"{comp['ap']['mean_difference']:+.2f}, interval includes 0")
    for confuser in ("stroke", "impasto"):
        fa = fused[f"{confuser}_false_alarm"]
        if fa > FALSE_ALARM_LIMIT:
            raise CheckFailed(f"AND-fusion calls {fa:.0%} of {confuser} pixels crack "
                              f"(limit {FALSE_ALARM_LIMIT:.0%})")
    return (f"AND-fusion AP {fused['ap']:.2f}, false alarms: strokes {fused['stroke_false_alarm']:.0%}, "
            f"impasto {fused['impasto_false_alarm']:.0%}")


def _stress(ctx: Context) -> str:
    truths = [s.crack_mask for _, s in ctx.eval_set]
    for k in range(0, 5):
        shifted = [replace(s, height=np.roll(s.height, k, axis=1)) for _, s in ctx.eval_set]
        ctx.shifts[k] = tolerant_pr_curve([score_fused(s) for s in shifted], truths).average_precision
        ctx.say(f"height map shifted {k} px: AND-fusion AP {ctx.shifts[k]:.2f}")
    if ctx.shifts[SHIFT_BUDGET_PX] < 0.9:
        raise CheckFailed(f"AND-fusion AP falls to {ctx.shifts[SHIFT_BUDGET_PX]:.2f} at {SHIFT_BUDGET_PX} px shift")
    return "AND-fusion AP by shift: " + ", ".join(f"{k}px {v:.2f}" for k, v in ctx.shifts.items())


GROUPS = ("Phase 1 · Make the data", "Phase 2 · Check the data", "Phase 3 · Score every pixel", "Phase 4 · Measure")

STEPS: tuple[Step, ...] = (
    Step("gen", "1 · Generate one sample",
         ("The recipe: V-shaped cracks, raised paint strokes, impasto valleys with rims, noise, misalignment.",
          "Generate seed 3000 twice.",
          "Same seed must give the same pixels; this is what makes every number repeatable."),
         "generate.py · generate_sample", "same seed, same pixels; the sample contains a crack",
         _generate, GROUPS[0]),
    Step("rebuild", "2 · Rebuild stored samples",
         ("The stored evaluation set was built from seeds. Rebuild its first 3 samples.",
          "Compare pixels with what is stored on disk.",
          "Pixels, not file bytes: PNG bytes can differ between computers for identical pixels."),
         "generate.py · generate_sample, load_sample", "every rebuilt sample matches its stored pixels",
         _rebuild, GROUPS[0]),
    Step("check", "3 · Check the fixed data",
         ("Two fixed sets: 30 validation samples choose settings, 30 evaluation samples report results.",
          "When they were built, every file got a fingerprint (sha256).",
          "Recompute every fingerprint and compare. One changed byte is enough to fail."),
         "eval_set.py · verify_eval_set", "every file matches its fingerprint", _check_data, GROUPS[1]),
    Step("load", "4 · Load and sanity-check",
         ("Read each sample from disk: photo, height map, crack, stroke and impasto masks.",
          "Check the sizes agree and that each sensor actually measured something.",
          "Bad input is stopped here, before any model sees it."),
         "generate.py · load_sample", "same size everywhere; photo and height both vary; every sample has a crack",
         _load, GROUPS[1]),
    Step("score", "5 · Classical models",
         ("One detector (black top-hat): 'thin, and darker or lower than its surroundings'.",
          "Four models: colour only, height only, OR (either sensor), AND (both sensors).",
          "Each model gives every pixel a score from 0 to 1."),
         "baseline.py · MODELS", "every score map has the sample's size and stays within 0-1", _score, GROUPS[2]),
    Step("unet", "6 · U-Net models",
         ("A small neural network learns the crack rule from freshly generated training samples.",
          "Two versions: photo only, and photo plus height map.",
          "Trained on seeds 10000+, which the evaluation never sees."),
         "unet.py · UNet, train_unet.py", "both checkpoints load and score every pixel (waits until they are trained)",
         _unet, GROUPS[2]),
    Step("cutoffs", "7 · Choose cutoffs (validation only)",
         ("A score becomes yes/no at a cutoff.",
          f"For each model, take the strictest cutoff that still finds {TARGET_RECALL:.0%} of cracks.",
          "Chosen on validation samples only, so the evaluation set stays unseen."),
         "evaluate.py · choose_thresholds", f"every model reaches {TARGET_RECALL:.0%} recall on validation",
         _cutoffs, GROUPS[3]),
    Step("measure", "8 · Measure on unseen samples",
         ("Apply each cutoff to the 30 evaluation samples.",
          "Measure AP, cracks found, strokes called crack, impasto called crack.",
          "Compare AND with every other model sample by sample, with 95% intervals."),
         "evaluate.py · per_sample_rows, paired_comparison",
         f"AND beats colour-only, height-only and OR on AP (interval above 0), under {FALSE_ALARM_LIMIT:.0%} false alarms; U-Nets reported",
         _measure, GROUPS[3]),
    Step("stress", "9 · Stress test: alignment",
         ("Real scanners never line the photo and height map up perfectly.",
          "Shift the height map 0-4 px against the photo and measure AND-fusion again.",
          "Shows how much alignment error the method can take."),
         "pipeline.py · _stress", f"AND-fusion keeps AP at or above 0.9 up to {SHIFT_BUDGET_PX} px", _stress,
         GROUPS[3]),
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

    broken = False  # a failure stops the run: later steps are skipped
    paused = False  # a waiting phase holds the run: later steps are held until it finishes
    for step, result in zip(STEPS, results):
        if broken:
            result.status = "skipped"
            continue
        if paused:
            result.status = "held"
            continue
        result.status = "running"

        def say(msg: str, r: StepResult = result) -> None:
            r.log.append(msg)
            notify(results)

        ctx.say = say
        notify(results)
        start = time.perf_counter()
        try:
            result.summary = step.run(ctx)
            result.status = "passed"
        except NotReady as e:  # the phase is not finished yet: pause here, do not run the next phase
            result.status, result.summary = "waiting", str(e)
            paused = True
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
