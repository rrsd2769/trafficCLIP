"""
Render a grid of sample flow images, N per class, to check that classes look
distinct and that the 54-byte header mask landed (the top-left of every tile
should be black).

    PYTHONPATH=. .venv/bin/python scripts/preview_samples.py
    PYTHONPATH=. .venv/bin/python scripts/preview_samples.py --npz data/processed/ustc_10class.npz
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()

    config = load_config()
    npz = Path(args.npz or config["paths"]["ustc_20class_file"])
    out = Path(args.out or f"results/figures/preview_{npz.stem}.png")

    with np.load(npz, allow_pickle=True) as d:
        x, y, labels = d["x"], d["y"], list(d["labels"])

    rng = np.random.default_rng(args.seed)
    n_rows, n_cols = len(labels), args.per_class

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.45), squeeze=False
    )
    for r, name in enumerate(labels):
        idx = np.flatnonzero(y == r)
        picks = rng.choice(idx, size=min(n_cols, len(idx)), replace=False)
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(picks):
                ax.imshow(x[picks[c]][0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)

    fig.suptitle(
        f"{npz.name}: {n_cols} samples/class (top-left 54 bytes masked)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
