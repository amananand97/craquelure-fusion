"""Presentation app: the whole pipeline, one sample at a time, then the results.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craquelure_fusion.baseline import predict_colour_only, predict_fused  # noqa: E402
from craquelure_fusion.eval_set import verify_eval_set  # noqa: E402
from craquelure_fusion.generate import load_sample  # noqa: E402
from craquelure_fusion.metrics import stroke_false_alarm_rate, tolerant_precision_recall  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
RESULTS = ROOT / "results" / "results.csv"

PIPELINE = """
flowchart LR
  subgraph GEN["1. Generate (synthetic, ground truth known)"]
    G["generate.py<br/>one fake painting"] --> RGB["rgb.png"]
    G --> H["height.png"]
    G --> M["mask.png<br/>crack = answer"]
    G --> S["stroke_mask.png<br/>confuser"]
  end
  subgraph MODELS["2. Models"]
    RGB --> A["Model A<br/>colour only"]
    RGB --> B["Model B<br/>colour + height"]
    H --> B
  end
  subgraph EVAL["3. Evaluate on the FIXED set"]
    A --> E["compare with mask.png"]
    B --> E
    M --> E
    S --> E
    E --> R["precision, recall,<br/>stroke false alarms"]
  end
"""


def _sample_ids() -> list[str]:
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and p.name.isdigit())


def page_pipeline() -> None:
    st.header("The pipeline")
    st.write("Every number on the next pages comes from the boxes below, in this order.")
    components.html(
        f"""
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
        </script>
        <pre class="mermaid">{PIPELINE}</pre>
        """,
        height=420,
    )
    problems = verify_eval_set(EVAL_DIR)
    if problems:
        st.error("The evaluation set does not match its manifest: " + "; ".join(problems[:3]))
    else:
        st.success(f"Evaluation set verified: {len(_sample_ids())} samples, unchanged since they were built.")


def page_one_sample() -> None:
    st.header("One sample, both models")
    sample_id = st.selectbox("Sample", _sample_ids())
    sample = load_sample(EVAL_DIR / sample_id)
    pred_a = predict_colour_only(sample.rgb)
    pred_b = predict_fused(sample.rgb, sample.height)

    panels = [
        ("rgb (photo)", sample.rgb),
        ("height", sample.height),
        ("answer: crack", sample.crack_mask),
        ("confuser: stroke", sample.stroke_mask),
        ("Model A prediction", pred_a),
        ("Model B prediction", pred_b),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, img) in zip(axes.flat, panels):
        if img.ndim == 2 and img.dtype == bool:
            ax.imshow(img, cmap="gray")
        elif img.ndim == 2:
            ax.imshow(img, cmap="viridis")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col_a, col_b = st.columns(2)
    for col, name, pred in ((col_a, "Model A (colour only)", pred_a), (col_b, "Model B (colour + height)", pred_b)):
        precision, recall = tolerant_precision_recall(pred, sample.crack_mask)
        fa = stroke_false_alarm_rate(pred, sample.crack_mask, sample.stroke_mask)
        col.metric(f"{name}: stroke false alarms", f"{fa:.0%}")
        col.caption(f"precision {precision:.2f} · recall {recall:.2f}")


def page_results() -> None:
    st.header("Results on the fixed set")
    if not RESULTS.exists():
        st.warning(f"Run `python -m craquelure_fusion.evaluate` first. Missing: {RESULTS.name}")
        return
    with open(RESULTS, newline="") as f:
        rows = list(csv.DictReader(f))
    models = sorted({r["model"] for r in rows})
    means = {
        m: {k: float(np.mean([float(r[k]) for r in rows if r["model"] == m]))
            for k in ("precision", "recall", "stroke_false_alarm")}
        for m in models
    }
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        col.metric(m, f"{means[m]['stroke_false_alarm']:.0%} stroke false alarms",
                   f"precision {means[m]['precision']:.2f} · recall {means[m]['recall']:.2f}",
                   delta_color="off")
    st.write("Lower stroke false alarms is better. Recall should stay high, or the model is just predicting less.")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(models, [means[m]["stroke_false_alarm"] for m in models], color=["#b8a88a", "#5b7f5b"])
    ax.set_ylabel("share of stroke pixels called crack")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Craquelure fusion", layout="wide")
    st.title("Craquelure fusion: does height stop paint being called crack?")
    page = st.sidebar.radio("Page", ["Pipeline", "One sample", "Results"])
    if page == "Pipeline":
        page_pipeline()
    elif page == "One sample":
        page_one_sample()
    else:
        page_results()


main()
