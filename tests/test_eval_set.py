"""The fixed evaluation set must be built the same way every time and be checkable."""

import csv
from pathlib import Path

from craquelure_fusion.eval_set import FILES_PER_SAMPLE, build_eval_set, verify_eval_set
from craquelure_fusion.generate import SampleConfig

SMALL = SampleConfig(size=64)


def test_builds_every_file_for_every_sample(tmp_path: Path) -> None:
    out = build_eval_set(tmp_path / "eval", n=2, base_cfg=SMALL)
    for sample_id in ("000", "001"):
        for name in FILES_PER_SAMPLE:
            assert (out / sample_id / name).exists(), f"missing {sample_id}/{name}"
    with open(out / "manifest.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_same_seeds_give_identical_bytes(tmp_path: Path) -> None:
    a = build_eval_set(tmp_path / "a", n=2, base_cfg=SMALL)
    b = build_eval_set(tmp_path / "b", n=2, base_cfg=SMALL)
    assert (a / "manifest.csv").read_text() == (b / "manifest.csv").read_text()


def test_fresh_build_verifies_clean(tmp_path: Path) -> None:
    out = build_eval_set(tmp_path / "eval", n=2, base_cfg=SMALL)
    assert verify_eval_set(out) == []


def test_changed_file_is_reported(tmp_path: Path) -> None:
    out = build_eval_set(tmp_path / "eval", n=2, base_cfg=SMALL)
    (out / "001" / "mask.png").write_bytes(b"tampered")
    assert verify_eval_set(out) == [f"changed {out / '001' / 'mask.png'}"]
