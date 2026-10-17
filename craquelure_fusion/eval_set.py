"""The FIXED evaluation set: built once, saved to disk, committed, never regenerated silently.

Why fixed: every model we compare must be scored on exactly the same pixels.
Why seeds 1000+: training data (later) will use other seeds, so no sample is
ever seen during training and then scored.

Build it:
    python -m craquelure_fusion.eval_set --out data/eval
Check it has not changed since it was built:
    python -m craquelure_fusion.eval_set --out data/eval --verify
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
from pathlib import Path

from craquelure_fusion.generate import SampleConfig, generate_sample, save_sample

logger = logging.getLogger(__name__)

EVAL_SEED_START = 1000   # eval seeds are reserved; training will never use them
EVAL_N_SAMPLES = 30
FILES_PER_SAMPLE = ("rgb.png", "height.png", "mask.png", "stroke_mask.png")
MANIFEST = "manifest.csv"
CONFIG = "config.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_eval_set(
    out_dir: Path,
    n: int = EVAL_N_SAMPLES,
    seed_start: int = EVAL_SEED_START,
    base_cfg: SampleConfig | None = None,
) -> Path:
    """Generate n samples with seeds seed_start..seed_start+n-1 and save them.

    Writes out_dir/000/, out_dir/001/, ... plus manifest.csv (one row per sample,
    with a sha256 of every file) and config.json (the settings used).
    """
    base = base_cfg or SampleConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(n):
        seed = seed_start + i
        sample = generate_sample(dataclasses.replace(base, seed=seed))
        sample_id = f"{i:03d}"
        sample_dir = out_dir / sample_id
        save_sample(sample, sample_dir)
        row = {
            "id": sample_id,
            "seed": seed,
            "crack_px": int(sample.crack_mask.sum()),
            "stroke_px": int(sample.stroke_mask.sum()),
            "overlap_px": int((sample.crack_mask & sample.stroke_mask).sum()),
        }
        for name in FILES_PER_SAMPLE:
            row[f"sha256_{name}"] = _sha256(sample_dir / name)
        rows.append(row)

    with open(out_dir / MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / CONFIG).write_text(json.dumps(dataclasses.asdict(base), indent=2) + "\n")
    logger.info("Built eval set: %d samples in %s", n, out_dir)
    return out_dir


def verify_eval_set(out_dir: Path) -> list[str]:
    """Return a list of problems. An empty list means the set is exactly as built."""
    problems: list[str] = []
    manifest_path = out_dir / MANIFEST
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]

    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            for name in FILES_PER_SAMPLE:
                path = out_dir / row["id"] / name
                if not path.exists():
                    problems.append(f"missing {path}")
                elif _sha256(path) != row[f"sha256_{name}"]:
                    problems.append(f"changed {path}")
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or verify the fixed evaluation set.")
    parser.add_argument("--out", type=Path, default=Path("data/eval"))
    parser.add_argument("--n", type=int, default=EVAL_N_SAMPLES)
    parser.add_argument("--verify", action="store_true", help="check files against manifest.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.verify:
        problems = verify_eval_set(args.out)
        if problems:
            for p in problems:
                print(p)
            raise SystemExit(1)
        print(f"OK: {args.out} matches its manifest")
        return

    build_eval_set(args.out, n=args.n)
    print(f"Built {args.n} samples in {args.out}")


if __name__ == "__main__":
    main()
