"""FIXED data sets: built once, saved, committed, never regenerated silently.

Two splits, with seeds that never overlap:
    val   seeds 2000+  -> used to CHOOSE thresholds
    eval  seeds 3000+  -> used ONLY to report the final numbers
Choosing a threshold on the same samples you report would flatter every model.
Seeds 1000+ were the v1 eval set; the v2 design was made while looking at those
images, so v2 is reported on seeds nobody has looked at (3000+).

Versions (the data design, not the code):
    v1  crack + painted stroke                  (committed in data/eval_v1, kept for history)
    v2  crack + painted stroke + impasto valley (the honest test: each single sensor has a confuser)

Build:   python -m craquelure_fusion.eval_set --version v2 --split eval
Verify:  python -m craquelure_fusion.eval_set --version v2 --split eval --verify
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

VERSIONS = {
    "v1": SampleConfig(),
    "v2": SampleConfig(n_impasto=1),
}
SEED_START = {"eval": 3000, "val": 2000, "eval_v1": 1000}
N_SAMPLES = 30
FILES_PER_SAMPLE = ("rgb.png", "height.png", "mask.png", "stroke_mask.png", "impasto_mask.png")
MANIFEST = "manifest.csv"
CONFIG = "config.json"


def default_dir(version: str, split: str) -> Path:
    return Path("data") / f"{split}_{version}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_eval_set(
    out_dir: Path,
    n: int = N_SAMPLES,
    seed_start: int = SEED_START["eval"],
    base_cfg: SampleConfig | None = None,
) -> Path:
    """Generate n samples with seeds seed_start .. seed_start+n-1 and save them.

    Writes out_dir/000/, out_dir/001/, ... plus manifest.csv (one row per sample,
    with a sha256 of every file) and config.json (the settings used).
    """
    base = base_cfg or VERSIONS["v2"]
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
            "impasto_px": int(sample.impasto_mask.sum()),
        }
        for name in FILES_PER_SAMPLE:
            row[f"sha256_{name}"] = _sha256(sample_dir / name)
        rows.append(row)

    with open(out_dir / MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / CONFIG).write_text(json.dumps(dataclasses.asdict(base), indent=2) + "\n")
    logger.info("Built %d samples in %s", n, out_dir)
    return out_dir


def verify_eval_set(out_dir: Path) -> list[str]:
    """Return a list of problems. Empty list = every file is exactly as built.

    The files to check are read from the manifest's sha256_* columns, so an
    older set (v1 has no impasto file) verifies against what it actually has.
    """
    manifest_path = out_dir / MANIFEST
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]

    problems: list[str] = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        hashed = [c.removeprefix("sha256_") for c in reader.fieldnames if c.startswith("sha256_")]
        for row in reader:
            for name in hashed:
                path = out_dir / row["id"] / name
                if not path.exists():
                    problems.append(f"missing {path}")
                elif _sha256(path) != row[f"sha256_{name}"]:
                    problems.append(f"changed {path}")
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or verify a fixed data set.")
    parser.add_argument("--version", choices=sorted(VERSIONS), default="v2")
    parser.add_argument("--split", choices=["eval", "val"], default="eval")
    parser.add_argument("--out", type=Path, default=None, help="default: data/<split>_<version>")
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--verify", action="store_true", help="check files against manifest.csv")
    args = parser.parse_args(argv)
    out = args.out or default_dir(args.version, args.split)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.verify:
        problems = verify_eval_set(out)
        for p in problems:
            print(p)
        if problems:
            raise SystemExit(1)
        print(f"OK: {out} matches its manifest")
        return

    if (out / MANIFEST).exists():
        raise SystemExit(f"{out} already exists. A fixed set is never rebuilt in place; "
                         "delete it on purpose first, or build a new version.")
    seed_start = SEED_START["eval_v1"] if (args.version, args.split) == ("v1", "eval") else SEED_START[args.split]
    build_eval_set(out, n=args.n, seed_start=seed_start, base_cfg=VERSIONS[args.version])
    print(f"Built {args.n} samples in {out}")


if __name__ == "__main__":
    main()
