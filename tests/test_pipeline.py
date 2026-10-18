"""The nine-step pipeline: green when clean, and each demo fault stops the step built to catch it."""

from pathlib import Path

import dataclasses

import numpy as np
import pytest

from craquelure_fusion import pipeline
from craquelure_fusion.pipeline import FAULTS, STEPS, _where, run_pipeline

DATA = Path(__file__).resolve().parents[1] / "data"
EXPECTED_FAILURE = {
    "tampered_file": "check",
    "height_sensor_off": "load",
    "model_bug": "score",
    "misaligned_6px": "measure",
}


def test_waiting_phase_pauses_the_run_and_holds_later_phases(tmp_path: Path) -> None:
    # No U-Net checkpoints beside this data: phase 3 waits, so phase 4 must not run.
    # The data is linked into a folder with no checkpoints/, so the result does not depend on what is trained.
    (tmp_path / "data").symlink_to(DATA)
    results, _ = run_pipeline(tmp_path / "data")
    by_key = {r.key: r.status for r in results}
    assert all(by_key[k] == "passed" for k in ("gen", "rebuild", "check", "load", "score"))
    assert by_key["unet"] == "waiting"
    assert all(by_key[k] == "held" for k in ("cutoffs", "measure", "stress"))


@pytest.fixture
def unet_done(monkeypatch):
    """Phase 3 finished (stubbed). Used by the fault tests, which need phase 4 to be reached."""
    def finished(ctx):
        ctx.say("stub: U-Net assumed finished")
        return "stub: U-Net assumed finished"

    stubbed = tuple(dataclasses.replace(s, run=finished) if s.key == "unet" else s for s in STEPS)
    monkeypatch.setattr(pipeline, "STEPS", stubbed)


def test_clean_run_passes_every_step_once_phase_3_is_done(unet_done) -> None:
    results, ctx = run_pipeline(DATA)
    assert [r.status for r in results] == ["passed"] * len(STEPS)
    assert ctx.means["colour_and_height"]["ap"] > 0.9


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_each_fault_fails_its_step_and_skips_the_rest(fault: str, unet_done) -> None:
    results, _ = run_pipeline(DATA, fault)
    keys = [r.key for r in results]
    failed_at = keys.index(EXPECTED_FAILURE[fault])
    assert all(r.status in ("passed", "waiting") for r in results[:failed_at])
    assert results[failed_at].status == "failed"
    assert results[failed_at].error and results[failed_at].where
    assert all(r.status == "skipped" for r in results[failed_at + 1:])


def test_faults_never_touch_committed_data(unet_done) -> None:
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


def test_generate_and_rebuild_steps_pass_on_clean_data(unet_done) -> None:
    results, _ = run_pipeline(DATA)
    by_key = {r.key: r for r in results}
    assert by_key["gen"].status == "passed" and by_key["rebuild"].status == "passed"


def test_each_step_records_what_it_did(unet_done) -> None:
    results, _ = run_pipeline(DATA)
    for r in results:
        if r.status != "held":  # held steps did not run, so they have nothing to record
            assert r.log, f"step {r.key} recorded nothing"
