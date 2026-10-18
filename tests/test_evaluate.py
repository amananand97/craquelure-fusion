"""The whole experiment runs end to end on tiny sets and writes every output."""

import json
from pathlib import Path

from craquelure_fusion.baseline import MODELS
from craquelure_fusion.eval_set import build_eval_set
from craquelure_fusion.evaluate import run
from craquelure_fusion.generate import SampleConfig


def test_experiment_runs_end_to_end(tmp_path: Path) -> None:
    cfg = SampleConfig(size=96)
    val = build_eval_set(tmp_path / "val", n=3, seed_start=2000, base_cfg=cfg)
    ev = build_eval_set(tmp_path / "eval", n=3, seed_start=3000, base_cfg=cfg)
    # Classical models only: the test checks the protocol, not whichever checkpoints are on disk.
    summary = run(val, ev, tmp_path / "results", models=dict(MODELS))
    for name in ("per_sample.csv", "summary.json", "pr_curves.png", "misregistration.png"):
        assert (tmp_path / "results" / name).exists(), name
    saved = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert set(saved["means"]) == set(MODELS)
    assert summary["protocol"]["target_recall"] == 0.9


def test_paired_comparison_matches_by_sample_not_row_order() -> None:
    from craquelure_fusion.evaluate import paired_comparison

    rows = [{"sample": "001", "model": "a", "ap": 0.2}, {"sample": "000", "model": "a", "ap": 0.4},
            {"sample": "000", "model": "b", "ap": 0.5}, {"sample": "001", "model": "b", "ap": 0.9}]
    # pairs: 000 -> 0.5-0.4 = 0.1, 001 -> 0.9-0.2 = 0.7
    assert abs(paired_comparison(rows, "a", "b", "ap")["mean_difference"] - 0.4) < 1e-9
