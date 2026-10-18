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


def test_clean_run_stops_at_the_impasto_finding(unet_done) -> None:
    # On this data AND-fusion leaks on impasto (the valleys carry a little shadow), so the
    # false-alarm check fails. That is the finding: everything before it passes, and the
    # numbers are still computed in full for the page to show.
    results, ctx = run_pipeline(DATA)
    by_key = {r.key: r for r in results}
    assert all(by_key[k].status == "passed" for k in ("gen", "rebuild", "check", "load", "score", "unet", "cutoffs"))
    assert by_key["measure"].status == "failed" and "impasto" in by_key["measure"].error
    assert by_key["stress"].status == "skipped"
    assert ctx.means["colour_and_height"]["ap"] > 0.9
    assert len(ctx.comparisons) == len(ctx.means) - 1


def test_misalignment_fails_the_ap_check_not_just_the_impasto_limit(unet_done) -> None:
    results, _ = run_pipeline(DATA, "misaligned_6px")
    measure = next(r for r in results if r.key == "measure")
    assert measure.status == "failed" and "not clearly better" in measure.error


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
    assert verify_eval_set(DATA / "eval") == []


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
        if r.status not in ("held", "skipped"):  # these did not run, so they have nothing to record
            assert r.log, f"step {r.key} recorded nothing"
