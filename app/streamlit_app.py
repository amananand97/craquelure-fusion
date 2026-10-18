"""Interactive presentation: run the pipeline, watch each step turn green or red,
open a step to see what it does and what it did, focus on one phase, and explore
the results (models, samples, misalignment).

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craquelure_fusion.baseline import score_fused  # noqa: E402
from craquelure_fusion.evaluate import FUSED  # noqa: E402
from craquelure_fusion.metrics import tolerant_pr_curve  # noqa: E402
from craquelure_fusion.pipeline import FAULTS, GROUPS, STEPS, StepResult, run_pipeline  # noqa: E402

DATA = ROOT / "data"
COLOURS = {"pending": "#8a8a8a", "running": "#d49a00", "passed": "#2e7d32", "failed": "#c62828",
           "skipped": "#8a8a8a", "waiting": "#6d6d9e", "held": "#8a8a8a"}
ICONS = {"pending": "○", "running": "◔", "passed": "✓", "failed": "✗", "skipped": "–", "waiting": "⧗", "held": "⏸"}
LABELS = {"pending": "not run", "running": "running…", "passed": "passed", "failed": "failed",
          "skipped": "skipped", "waiting": "waiting", "held": "held"}
STATE_TEXT = {"pending": "not started", "running": "running…", "passed": "all passed", "failed": "failed",
              "skipped": "skipped", "waiting": "waiting for a trained model", "held": "held"}
NICE = {"colour_only": "Colour only", "height_only": "Height only",
        "colour_or_height": "Colour OR height", "colour_and_height": "Colour AND height",
        "unet_3ch": "U-Net: photo only", "unet_4ch": "U-Net: photo + height"}
SHORT = dict(zip(GROUPS, ("1 · Make data", "2 · Check data", "3 · Score pixels", "4 · Measure")))
BLURB = {
    GROUPS[0]: "Build the synthetic photos and height maps from seeds, so every number can be repeated exactly.",
    GROUPS[1]: "Confirm the fixed sets are unchanged and readable before any model looks at them.",
    GROUPS[2]: "Each detector gives every pixel a crack score from 0 to 1: classical rules first, then the U-Nets.",
    GROUPS[3]: "Choose each cutoff on validation samples, then measure on the 30 samples nobody tuned on.",
}
ALL = "All phases"


# --- helpers -------------------------------------------------------------------

def _plain(title: str) -> str:
    """'5 · Measure on unseen samples' -> 'Measure on unseen samples' (for sentences)."""
    return title.split(" · ", 1)[-1]


def _name(model: str) -> str:
    return NICE.get(model, model)


def _results() -> list[StepResult]:
    return st.session_state.get("results") or [StepResult(s.key, s.title) for s in STEPS]


def _pairs(results: list[StepResult], group: str) -> list[tuple]:
    return [(s, r) for s, r in zip(STEPS, results) if s.group == group]


def _phase_state(pairs: list[tuple]) -> str:
    """One state for a whole phase: the most urgent state among its steps."""
    states = [r.status for _, r in pairs]
    for urgent in ("failed", "running", "waiting", "held"):
        if urgent in states:
            return urgent
    if all(s == "passed" for s in states):
        return "passed"
    if all(s == "pending" for s in states):
        return "pending"
    return "skipped"


def _box_css(results: list[StepResult]) -> None:
    rules = []
    for r in results:
        style = "dashed" if r.status in ("skipped", "waiting", "held") else "solid"
        width = "3px" if r.status in ("passed", "failed", "running") else "2px"
        rules.append(f".st-key-step_{r.key} {{ border: {width} {style} {COLOURS[r.status]} !important; "
                     f"border-radius: 12px; padding: 0.9rem 1rem; min-height: 150px; }}")
    st.markdown(f"<style>{' '.join(rules)}</style>", unsafe_allow_html=True)


def _live_grid_html(results: list[StepResult]) -> str:
    """A plain picture of the grid, redrawn while the pipeline runs."""
    cells = []
    for r in results:
        c = COLOURS[r.status]
        style = "dashed" if r.status in ("skipped", "waiting", "held") else "solid"
        detail = html.escape(r.log[-1] if r.status == "running" and r.log else (r.summary or r.error or LABELS[r.status]))
        cells.append(
            f"<div style='border:3px {style} {c};border-radius:12px;padding:14px;min-height:120px'>"
            f"<div style='font-weight:600'>{html.escape(r.title)}</div>"
            f"<div style='color:{c};font-weight:600;margin:6px 0'>{ICONS[r.status]} {LABELS[r.status]}</div>"
            f"<div style='font-size:0.85rem;opacity:0.8'>{detail[:140]}</div></div>")
    return ("<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px'>"
            + "".join(cells) + "</div>")


# --- phase strip and step grid ----------------------------------------------------

def _phase_strip(results: list[StepResult]) -> None:
    """Four phase tiles: the whole run at a glance, in the order the phases must happen."""
    for col, group in zip(st.columns(len(GROUPS)), GROUPS):
        pairs = _pairs(results, group)
        state = _phase_state(pairs)
        done = sum(r.status == "passed" for _, r in pairs)
        col.markdown(
            f"**{SHORT[group]}**  \n"
            f"<span style='color:{COLOURS[state]};font-weight:600'>{ICONS[state]} {STATE_TEXT[state]}</span>  \n"
            f"<span style='opacity:0.75'>{done} of {len(pairs)} steps</span>",
            unsafe_allow_html=True)


def _interactive_grid(results: list[StepResult], focus: str | None) -> None:
    _box_css(results)
    for group in GROUPS:
        if focus and group != focus:
            continue
        pairs = _pairs(results, group)
        st.subheader(group)
        st.caption(BLURB[group])
        for start in range(0, len(pairs), 3):
            _grid_row(pairs[start:start + 3])


def _grid_row(row: list[tuple]) -> None:
    for col, (step, r) in zip(st.columns(3, gap="medium"), row):
        with col.container(key=f"step_{step.key}"):
            st.markdown(f"**{step.title}**")
            timing = f" · {r.seconds:.1f} s" if r.status in ("passed", "failed") else ""
            st.markdown(f"<span style='color:{COLOURS[r.status]};font-weight:600'>"
                        f"{ICONS[r.status]} {LABELS[r.status]}{timing}</span>", unsafe_allow_html=True)
            if r.status == "running" and r.log:
                st.caption(f"now: {r.log[-1]}")
            elif r.summary:
                st.caption(r.summary)
            if r.error:
                st.caption(f":red[{r.error}]")
            with st.popover("What's inside", width="stretch"):
                st.markdown(f"#### {step.title}")
                for line in step.what:
                    st.markdown(f"- {line}")
                st.markdown(f"**Passes if:** {step.passes_if}")
                st.markdown(f"**Code:** `{step.code}`")
                if r.status == "failed":
                    st.error(r.error)
                    st.markdown("**Where it broke:**")
                    st.code("\n".join(r.where) or "(no location recorded)", language=None)
                elif r.status == "skipped":
                    st.info("Skipped: an earlier step failed, so this one would be built on something broken.")
                elif r.status == "held":
                    st.info("Held: an earlier phase is still waiting, so this phase has not started.")
            if r.log:
                with st.expander(f"What it did · {len(r.log)} lines", expanded=r.status == "running"):
                    st.code("\n".join(r.log), language=None)


# --- results ---------------------------------------------------------------------

def _step_sample(delta: int, n: int) -> None:
    st.session_state["sample_idx"] = (st.session_state.get("sample_idx", 0) + delta) % n


def _results_section(results: list[StepResult]) -> None:
    ctx = st.session_state.get("ctx")
    by_key = {r.key: r for r in results}
    st.divider()
    st.header("Results")
    if ctx is None:
        st.caption("Press **Run pipeline** to compute them.")
        return
    if by_key["measure"].status != "passed":
        failed = next((r for r in results if r.status == "failed"), None)
        st.warning(f"No results: {_plain(failed.title) if failed else 'a step'} failed. "
                   "Every result depends on the steps before it, so nothing after a failure is trusted.")
        return

    names = list(ctx.means)
    best = max(names, key=lambda n: ctx.means[n]["ap"])
    st.success(f"Best on the 30 unseen samples: **{_name(best)}**, AP {ctx.means[best]['ap']:.2f} "
               "(1.00 is perfect).")

    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        m = ctx.means[name]
        col.metric(_name(name), f"AP {m['ap']:.2f}")
        col.caption(f"cracks found {m['recall']:.0%}  \nstrokes called crack {m['stroke_false_alarm']:.0%}  \n"
                    f"impasto called crack {m['impasto_false_alarm']:.0%}")

    left, right = st.columns(2)
    shown = left.multiselect("Models on the curve", names, default=names, format_func=_name, key="curve_models")
    truths = [s.crack_mask for _, s in ctx.eval_set]
    if shown:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name in shown:
            c = tolerant_pr_curve(ctx.eval_scores[name], truths)
            ax.plot(c.recall, c.precision, label=f"{_name(name)} (AP {c.average_precision:.2f})")
        ax.set(xlabel="cracks found (recall)", ylabel="predictions that are cracks (precision)",
               xlim=(0, 1.02), ylim=(0, 1.02))
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        left.pyplot(fig)
        plt.close(fig)
    else:
        left.info("Pick at least one model to draw its curve.")

    if ctx.shifts:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(list(ctx.shifts), list(ctx.shifts.values()), marker="o", color="#2e7d32")
        ax.set(xlabel="height map shifted against photo (px)", ylabel="AND-fusion AP", ylim=(0, 1.02),
               title="Alignment limits fusion")
        fig.tight_layout()
        right.pyplot(fig)
        plt.close(fig)

    st.subheader("Colour AND height against each other model")
    st.caption("Mean difference over the 30 samples. Positive AP = AND is better. "
               "The bracket is a 95% bootstrap interval: if it crosses 0, the gap could be chance.")
    lines = ["| compared with | AP | strokes called crack | impasto called crack |", "|---|---|---|---|"]
    for other, comp in ctx.comparisons.items():
        cells = [f"{c['mean_difference']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]" for c in comp.values()]
        lines.append(f"| {_name(other)} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))

    # One sample at a time: buttons move through the set, the slider jumps anywhere.
    st.subheader("Look at one sample")
    ids = [i for i, _ in ctx.eval_set]
    n = len(ids)
    prev_col, mid_col, next_col = st.columns([1, 4, 1])
    prev_col.button("◀ previous", on_click=_step_sample, args=(-1, n), use_container_width=True)
    next_col.button("next ▶", on_click=_step_sample, args=(1, n), use_container_width=True)
    idx = st.slider("Evaluation sample", 0, n - 1, key="sample_idx", format="%d")
    mid_col.caption(f"Sample {ids[idx]} · {idx + 1} of {n} unseen samples")
    s = ctx.eval_set[idx][1]

    panels = [("photo", s.rgb, None), ("height", s.height, "viridis"), ("answer: crack", s.crack_mask, "gray"),
              ("stroke (fools colour)", s.stroke_mask, "gray"), ("impasto (fools height)", s.impasto_mask, "gray")]
    panels += [(f"{_name(m)} · calls crack", ctx.eval_scores[m][idx] >= ctx.thresholds[m], "gray") for m in ctx.means]
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Move the height map and watch the AND mask break. The cutoff stays the one set at 0 px.
    st.subheader("Move the height map")
    k = st.slider("Height map shifted by (px)", 0, 4, 0, key="shift_px")
    shifted = replace(s, height=np.roll(s.height, k, axis=1))
    pred = score_fused(shifted) >= ctx.thresholds[FUSED]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(s.crack_mask, cmap="gray")
    axes[0].set_title("answer: crack", fontsize=10)
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title(f"AND-fusion at {k} px shift", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Cutoff fixed at 0 px on validation ({ctx.thresholds[FUSED]:.2f}). "
               "Watch the white mask lose its shape as the height map drifts.")

    st.info("Limits: synthetic data where each confuser copies a crack exactly in one sensor, so an "
            "agreement rule is bound to win; a classical detector; no real sensor noise. "
            "The point shown is the mechanism, not real-world accuracy.")


@st.fragment(run_every=3)
def _live_training() -> None:
    """U-Net training, read from the progress files every 3 seconds (no button press needed)."""
    ckpt = DATA.parent / "checkpoints"
    st.subheader("Live: U-Net training")
    any_run = False
    for name, label in (("unet_3ch", "photo only (3 channels)"), ("unet_4ch", "photo + height (4 channels)")):
        if (ckpt / f"{name}.pt").exists():
            st.markdown(f"**{label}:** finished, checkpoint saved")
            continue
        prog = ckpt / f"{name}.progress.json"
        if not prog.exists():
            st.markdown(f"**{label}:** not started (or started before progress logging)")
            continue
        any_run = True
        p = json.loads(prog.read_text())
        st.markdown(f"**{label}:** step {p['step']} of {p['steps']} · loss {p['loss']:.4f} · "
                    f"{p['seeds_used']} fresh training samples so far · updated {p['updated']}")
        st.progress(min(p["step"] / p["steps"], 1.0))
    if not any_run:
        st.caption("No training running. Checkpoints appear here as they finish.")


# --- page ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Craquelure fusion", layout="wide")
    st.title("How the crack detector works, step by step")
    st.markdown("A crack is **dark AND sunken**. A painted stroke is only dark; an impasto valley is only "
                "sunken. Does combining a photo with a height map tell them apart?")
    st.markdown("**How to read this:** the run goes through four phases in order, and a phase cannot start "
                "until the one before it passes. Each box is a step. **What's inside** explains it; "
                "**What it did** shows its log. "
                "Colours: :green[green] passed · :red[red] failed · :violet[dashed purple] waiting for a "
                "trained model · :gray[dashed grey] held or skipped.")

    with st.sidebar:
        st.header("Demo: break something")
        choice = st.radio("Fault", ["None (run normally)", *FAULTS.values()], index=0)
        st.caption("Faults are applied in memory or on a temporary copy. The committed data is never touched.")
        st.markdown("**What each fault breaks:**  \n" + "  \n".join(
            f"- {text} → step {n}" for text, n in zip(FAULTS.values(), (1, 2, 3, 5))))
    fault = next((k for k, v in FAULTS.items() if v == choice), None)

    run_clicked = st.button("Run pipeline", type="primary")
    progress_slot = st.empty()
    _live_training()
    strip_slot = st.container()
    focus_slot = st.container()
    grid_slot = st.empty()

    if run_clicked:
        def on_update(results: list[StepResult]) -> None:
            done = sum(r.status in ("passed", "failed", "waiting") for r in results)
            running = next((_plain(r.title) for r in results if r.status == "running"), "finished")
            progress_slot.progress(done / len(results), text=f"{done} of {len(results)} steps · {running}")
            grid_slot.markdown(_live_grid_html(results), unsafe_allow_html=True)

        results, ctx = run_pipeline(DATA, fault, on_update)
        st.session_state.update(results=results, ctx=ctx, fault=fault)
        st.rerun()

    results = _results()
    done = sum(r.status in ("passed", "failed", "waiting") for r in results)
    failed = next((r for r in results if r.status == "failed"), None)
    waiting = [_plain(r.title) for r in results if r.status == "waiting"]
    text = ("not run yet" if done == 0 else
            f"stopped: {_plain(failed.title)} failed" if failed else
            f"paused: waiting for {', '.join(waiting)}" if waiting else "all steps passed")
    progress_slot.progress(done / len(results), text=f"{done} of {len(results)} steps · {text}")

    with strip_slot:
        _phase_strip(results)
    with focus_slot:
        focus_choice = st.segmented_control("Show", [ALL, *SHORT.values()], default=ALL, key="focus")
    focus = next((g for g, label in SHORT.items() if label == focus_choice), None)
    with grid_slot.container():
        _interactive_grid(results, focus)
    _results_section(results)


main()
