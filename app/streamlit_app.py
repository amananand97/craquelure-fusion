"""Interactive presentation: run the pipeline, watch each step turn green or red,
click a step to see what it does, see the results underneath.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craquelure_fusion.baseline import MODELS  # noqa: E402
from craquelure_fusion.evaluate import FUSED  # noqa: E402
from craquelure_fusion.metrics import tolerant_pr_curve  # noqa: E402
from craquelure_fusion.pipeline import FAULTS, STEPS, StepResult, run_pipeline  # noqa: E402

DATA = ROOT / "data"
COLOURS = {"pending": "#b8b8b8", "running": "#d49a00", "passed": "#2e7d32", "failed": "#c62828", "skipped": "#b8b8b8", "waiting": "#6d6d9e", "held": "#8a8a8a"}
ICONS = {"pending": "○", "running": "◔", "passed": "✓", "failed": "✗", "skipped": "–", "waiting": "⧗", "held": "⏸"}
LABELS = {"pending": "not run", "running": "running…", "passed": "passed", "failed": "failed", "skipped": "skipped", "waiting": "waiting", "held": "held"}
NICE = {"colour_only": "Colour only", "height_only": "Height only",
        "colour_or_height": "Colour OR height", "colour_and_height": "Colour AND height"}


# --- step grid -----------------------------------------------------------------

def _plain(title: str) -> str:
    """'5 · Measure on unseen samples' -> 'Measure on unseen samples' (for sentences)."""
    return title.split(" · ", 1)[-1]


def _results() -> list[StepResult]:
    return st.session_state.get("results") or [StepResult(s.key, s.title) for s in STEPS]


def _box_css(results: list[StepResult]) -> None:
    rules = []
    for r in results:
        style = "dashed" if r.status in ("skipped", "waiting", "held") else "solid"
        width = "3px" if r.status in ("passed", "failed", "running") else "2px"
        rules.append(f".st-key-step_{r.key} {{ border: {width} {style} {COLOURS[r.status]} !important; "
                     f"border-radius: 12px; padding: 0.9rem 1rem; min-height: 165px; }}")
    st.markdown(f"<style>{' '.join(rules)}</style>", unsafe_allow_html=True)


def _live_grid_html(results: list[StepResult]) -> str:
    """A plain picture of the grid, redrawn while the pipeline runs."""
    cells = []
    for r in results:
        c = COLOURS[r.status]
        style = "dashed" if r.status == "skipped" else "solid"
        detail = html.escape(r.log[-1] if r.status == "running" and r.log else (r.summary or r.error or LABELS[r.status]))
        cells.append(
            f"<div style='border:3px {style} {c};border-radius:12px;padding:14px;min-height:120px'>"
            f"<div style='font-weight:600'>{html.escape(r.title)}</div>"
            f"<div style='color:{c};font-weight:600;margin:6px 0'>{ICONS[r.status]} {LABELS[r.status]}</div>"
            f"<div style='font-size:0.85rem;opacity:0.8'>{detail[:140]}</div></div>")
    return ("<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px'>"
            + "".join(cells) + "</div>")


def _interactive_grid(results: list[StepResult]) -> None:
    _box_css(results)
    for group in dict.fromkeys(s.group for s in STEPS):  # groups in order of first appearance
        st.subheader(group)
        members = [s for s in STEPS if s.group == group]
        for start in range(0, len(members), 3):
            _grid_row(results, members[start:start + 3])


def _grid_row(results: list[StepResult], row: list) -> None:
    for col, step in zip(st.columns(3, gap="medium"), row):
        r = next(x for x in results if x.key == step.key)
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


# --- results ---------------------------------------------------------------------

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

    cols = st.columns(len(MODELS))
    for col, name in zip(cols, MODELS):
        m = ctx.means[name]
        col.metric(NICE[name], f"AP {m['ap']:.2f}")
        col.caption(f"cracks found {m['recall']:.0%}  \nstrokes called crack {m['stroke_false_alarm']:.0%}  \n"
                    f"impasto called crack {m['impasto_false_alarm']:.0%}")
    st.caption("AP = area under the precision-recall curve, mean over 30 unseen samples. 1.00 is perfect.")

    left, right = st.columns(2)
    truths = [s.crack_mask for _, s in ctx.eval_set]
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in MODELS:
        c = tolerant_pr_curve(ctx.eval_scores[name], truths)
        ax.plot(c.recall, c.precision, label=NICE[name])
    ax.set(xlabel="cracks found (recall)", ylabel="predictions that are cracks (precision)", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    left.pyplot(fig)
    plt.close(fig)

    if ctx.shifts:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(list(ctx.shifts), list(ctx.shifts.values()), marker="o", color="#2e7d32")
        ax.set(xlabel="height map shifted against photo (px)", ylabel="AND-fusion AP", ylim=(0, 1.02),
               title="Alignment limits fusion")
        fig.tight_layout()
        right.pyplot(fig)
        plt.close(fig)

    st.subheader("Colour AND height minus each other model, sample by sample")
    lines = ["| vs | AP | strokes called crack | impasto called crack |", "|---|---|---|---|"]
    for other, comp in ctx.comparisons.items():
        cells = [f"{c['mean_difference']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]" for c in comp.values()]
        lines.append(f"| {NICE[other]} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))
    st.caption("Mean difference over 30 samples, with a 95% bootstrap interval in brackets.")

    st.subheader("Look at one sample")
    ids = [i for i, _ in ctx.eval_set]
    idx = ids.index(st.selectbox("Evaluation sample", ids))
    s = ctx.eval_set[idx][1]
    panels = [("photo", s.rgb, None), ("height", s.height, "viridis"), ("answer: crack", s.crack_mask, "gray"),
              ("stroke (fools colour)", s.stroke_mask, "gray"), ("impasto (fools height)", s.impasto_mask, "gray")]
    panels += [(NICE[n], ctx.eval_scores[n][idx] >= ctx.thresholds[n], "gray") for n in MODELS]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

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

    with st.sidebar:
        st.header("Demo: break something")
        choice = st.radio("Fault", ["None (run normally)", *FAULTS.values()], index=0)
        st.caption("Faults are applied in memory or on a temporary copy. The committed data is never touched.")
        st.markdown("**What each fault breaks:**  \n" + "  \n".join(
            f"- {text} → step {n}" for text, n in zip(FAULTS.values(), (1, 2, 3, 5))))
    fault = next((k for k, v in FAULTS.items() if v == choice), None)

    run_clicked = st.button("Run pipeline", type="primary")
    progress_slot = st.empty()
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

    _live_training()
    results = _results()
    done = sum(r.status in ("passed", "failed", "waiting") for r in results)
    failed = next((r for r in results if r.status == "failed"), None)
    waiting = [_plain(r.title) for r in results if r.status == "waiting"]
    text = ("not run yet" if done == 0 else
            f"stopped: {_plain(failed.title)} failed" if failed else
            f"paused: waiting for {', '.join(waiting)}" if waiting else "all steps passed")
    progress_slot.progress(done / len(results), text=f"{done} of {len(results)} steps · {text}")
    with grid_slot.container():
        _interactive_grid(results)
    _results_section(results)


main()
