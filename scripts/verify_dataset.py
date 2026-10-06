"""
Check the built .npz archives against the documented schema.

    PYTHONPATH=. .venv/bin/python scripts/verify_dataset.py
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.make_traffic_dataset import HEADER_MASK_BYTES
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def check(archive: Path, expect_classes: int, expect_per_class: int) -> bool:
    if not archive.exists():
        logging.error(f"{archive} missing")
        return False

    ok = True
    with np.load(archive, allow_pickle=True) as d:
        x, y, m, labels = d["x"], d["y"], d["m"], d["labels"]

    n = expect_classes * expect_per_class
    print(f"\n=== {archive.name} ===")
    print(f"x      {x.shape} {x.dtype}")
    print(f"y      {y.shape} {y.dtype}")
    print(f"m      {m.shape} {m.dtype}")
    print(f"labels {labels.shape} -> {list(labels)}")

    def assert_(cond, msg):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and bool(cond)

    assert_(x.shape == (n, 1, 28, 28), f"x is ({n}, 1, 28, 28)")
    assert_(y.shape == (n,), f"y is ({n},)")
    assert_(m.shape == (n, 3), f"m is ({n}, 3)")
    assert_(labels.shape == (expect_classes,), f"labels is ({expect_classes},)")

    counts = Counter(y.tolist())
    balanced = len(counts) == expect_classes and set(counts.values()) == {expect_per_class}
    assert_(balanced, f"all {expect_classes} classes have exactly {expect_per_class} samples")
    if not balanced:
        print(f"        actual: {dict(sorted(counts.items()))}")

    assert_(np.isfinite(m).all(), "no NaN or Inf in the stats vectors")
    assert_(bool((x >= 0).all() and (x <= 1).all()), "images normalised to [0, 1]")

    masked = np.all(x.reshape(len(x), -1)[:, :HEADER_MASK_BYTES] == 0)
    assert_(masked, f"first {HEADER_MASK_BYTES} bytes zeroed in every sample")

    assert_(list(labels) == sorted(labels), "labels are in sorted order (deterministic)")

    nonconstant = sum(1 for i in range(len(x)) if x[i].std() > 0)
    assert_(nonconstant == len(x), "no all-zero images")

    print(f"  stats ranges: mean_IAT {m[:,0].min():.3f}..{m[:,0].max():.3f} ms, "
          f"jitter {m[:,1].min():.3f}..{m[:,1].max():.3f} ms, "
          f"entropy {m[:,2].min():.3f}..{m[:,2].max():.3f} bits")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument(
        "--variants",
        action="store_true",
        help="Also check the _t60 flow-timeout variants when present.",
    )
    args = parser.parse_args()

    config = load_config()
    per_class = args.samples_per_class or config["dataset"]["traffic"]["samples_per_class"]

    archives = [
        (Path(config["paths"]["ustc_10class_file"]), 10),
        (Path(config["paths"]["ustc_20class_file"]), 20),
    ]
    if args.variants:
        for base, k in list(archives):
            t60 = base.with_name(f"{base.stem}_t60{base.suffix}")
            if t60.exists():
                archives.append((t60, k))

    results = [check(a, k, per_class) for a, k in archives]
    print()
    if all(results):
        logging.info("DATASET CHECK PASSED")
        return 0
    logging.error("DATASET CHECK FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
