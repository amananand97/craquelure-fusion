"""The six-step pipeline: green when clean, and each demo fault stops the step built to catch it."""

from pathlib import Path

import numpy as np
import pytest

from craquelure_fusion.pipeline import FAULTS, STEPS, _where, run_pipeline

DATA = Path(__file__).resolve().parents[1] / "data"
EXPECTED_FAILURE = {
    "tampered_file": "check",
    "height_sensor_off": "load",
    "model_bug": "score",
    "misaligned_6px": "measure",
}


def test_clean_run_passes_every_step() -> None:
    results, ctx = run_pipeline(DATA)
    assert [r.status for r in results] == ["passed"] * len(STEPS)
    assert ctx.means["colour_and_height"]["ap"] > 0.9


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_each_fault_fails_its_step_and_skips_the_rest(fault: str) -> None:
    results, _ = run_pipeline(DATA, fault)
    keys = [r.key for r in results]
    failed_at = keys.index(EXPECTED_FAILURE[fault])
    assert [r.status for r in results[:failed_at]] == ["passed"] * failed_at
    assert results[failed_at].status == "failed"
    assert results[failed_at].error and results[failed_at].where
    assert all(r.status == "skipped" for r in results[failed_at + 1:])


def test_faults_never_touch_committed_data() -> None:
    from craquelure_fusion.eval_set import verify_eval_set

    run_pipeline(DATA, "tampered_file")
    assert verify_eval_set(DATA / "eval_v2") == []


def test_crash_location_points_into_our_code() -> None:
    from craquelure_fusion.metrics import tolerant_pr_curve

    with pytest.raises(Exception) as info:
        tolerant_pr_curve([np.zeros((3, 3))], [np.zeros((4, 4), dtype=bool)])
    assert any(line.startswith("metrics.py:") for line in _where(info.value))


def test_unknown_fault_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_pipeline(DATA, "no_such_fault")
