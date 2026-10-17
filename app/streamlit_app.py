"""Presentation: the question, the pipeline, one sample, the results.

Run from the repo root (after `python -m craquelure_fusion.evaluate`):
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craquelure_fusion.baseline import MODELS  # noqa: E402
from craquelure_fusion.eval_set import verify_eval_set  # noqa: E402
from craquelure_fusion.generate import load_sample  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval_v2"
VAL_DIR = ROOT / "data" / "val_v2"
RESULTS = ROOT / "results"

PIPELINE = """
flowchart LR
  subgraph DATA["1. Data (synthetic, answer known)"]
    G["generator"] --> RGB["photo (rgb)"]
    G --> H["height map"]
    G --> M["crack mask = answer"]
    G --> C["confuser masks:<br/>stroke (dark, flat)<br/>impasto (sunken, not dark)"]
  end
  subgraph MODELS["2. Three models, same detector"]
    RGB --> A["colour only"]
    H --> B["height only"]
    RGB --> F["colour AND height"]
    H --> F
  end
  subgraph EXP["3. Experiment"]
    V["val set: choose cutoff<br/>at 90% recall"] --> E
    A --> E["eval set: score"]
    B --> E
    F --> E
    M --> E
    C --> E
    E --> R["AP, false alarms per confuser,<br/>paired 95% intervals"]
    E --> S["stress test:<br/>shift height 0-4 px"]
  end
"""


def _ids() -> list[str]:
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and p.name.isdigit())


def _summary() -> dict | None:
    path = RESULTS / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def page_question() -> None:
    st.header("The question")
    st.markdown(
        "A crack in a painting is **dark AND sunken**. A painted brushstroke is dark but flat, "
        "so a camera alone is fooled by it. An impasto valley (between ridges of thick paint) is "
        "sunken but not dark, so a height scanner alone is fooled by it.\n\n"
        "**Claim tested:** combining an independent height map with the photo removes both kinds "
        "of false alarm, at the same crack recall."
    )
    components.html(
        f"""
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
        </script>
        <pre class="mermaid">{PIPELINE}</pre>
        """,
        height=460,
    )
    for d in (VAL_DIR, EVAL_DIR):
        problems = verify_eval_set(d)
        if problems:
            st.error(f"{d.name} does not match its manifest: " + "; ".join(problems[:3]))
        else:
            st.success(f"{d.name}: 30 samples, verified unchanged since they were built.")


def page_sample() -> None:
    st.header("One sample, three models")
    summary = _summary()
    if summary is None:
        st.warning("Run `python -m craquelure_fusion.evaluate` first.")
        return
    thresholds = summary["thresholds_from_val"]
    sample = load_sample(EVAL_DIR / st.selectbox("Eval sample", _ids()))

    panels = [
        ("photo", sample.rgb, None), ("height", sample.height, "viridis"),
        ("answer: crack", sample.crack_mask, "gray"),
        ("confuser: stroke (fools colour)", sample.stroke_mask, "gray"),
        ("confuser: impasto (fools height)", sample.impasto_mask, "gray"),
        ("", None, None),
    ] + [(f"{name} prediction", f(sample) >= thresholds[name], "gray") for name, f in MODELS.items()]
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        ax.axis("off")
        if img is not None:
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Cutoffs were chosen on the separate validation set, so every model finds about 90% of the crack.")


def page_results() -> None:
    st.header("Results on the fixed eval set (30 samples)")
    summary = _summary()
    if summary is None:
        st.warning("Run `python -m craquelure_fusion.evaluate` first.")
        return
    means = summary["means"]
    cols = st.columns(len(means))
    for col, (name, m) in zip(cols, means.items()):
        col.metric(name, f"AP {m['ap']:.2f}")
        col.caption(f"strokes called crack: {m['stroke_false_alarm']:.0%}  \n"
                    f"impasto called crack: {m['impasto_false_alarm']:.0%}  \n"
                    f"recall: {m['recall']:.0%}")

    left, right = st.columns(2)
    left.image(str(RESULTS / "pr_curves.png"), caption="Precision vs recall at every cutoff")
    right.image(str(RESULTS / "misregistration.png"), caption="Stress test: misaligned height map")

    st.subheader("Fused minus each single sensor, sample by sample")
    lines = []
    for other, comp in summary["fused_vs"].items():
        for metric, c in comp.items():
            p = "n/a" if c["wilcoxon_p"] is None else f"{c['wilcoxon_p']:.0e}"
            lines.append(f"| {other} | {metric} | {c['mean_difference']:+.3f} | "
                         f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {p} |")
    st.markdown("| vs | metric | mean difference | 95% interval | Wilcoxon p |\n|---|---|---|---|---|\n"
                + "\n".join(lines))
    st.info("Limits: synthetic data with clean, symmetric confusers; a classical detector; "
            "no real scanner noise yet. The fused AP near 1.0 says the data is easy, not that "
            "the problem is solved. Next: harder data, a learned model, and a real capture.")


def main() -> None:
    st.set_page_config(page_title="Craquelure fusion", layout="wide")
    st.title("Does a height map stop paint being called a crack?")
    page = st.sidebar.radio("Page", ["Question and pipeline", "One sample", "Results"])
    {"Question and pipeline": page_question, "One sample": page_sample, "Results": page_results}[page]()


main()
