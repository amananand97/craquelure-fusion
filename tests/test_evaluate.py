"""The whole experiment runs end to end on tiny sets and writes every output."""

import json
from pathlib import Path

from craquelure_fusion.eval_set import build_eval_set
from craquelure_fusion.evaluate import run
from craquelure_fusion.generate import SampleConfig


def test_experiment_runs_end_to_end(tmp_path: Path) -> None:
    cfg = SampleConfig(size=64, n_impasto=1)
    val = build_eval_set(tmp_path / "val", n=3, seed_start=2000, base_cfg=cfg)
    ev = build_eval_set(tmp_path / "eval", n=3, seed_start=1000, base_cfg=cfg)
    summary = run(val, ev, tmp_path / "results")
    for name in ("per_sample.csv", "summary.json", "pr_curves.png", "misregistration.png"):
        assert (tmp_path / "results" / name).exists(), name
    saved = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert set(saved["means"]) == {"colour_only", "height_only", "colour_plus_height"}
    assert summary["protocol"]["target_recall"] == 0.9
