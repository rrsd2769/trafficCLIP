"""Figures for Phase 4 (confusable-pair analysis):
  - per-class accuracy bar chart, 20 classes, Virut/Neris highlighted
  - boxplots of IAT/jitter/entropy for Virut vs Neris test samples, showing
    median overlap despite outlier-driven mean/std divergence

Reads results/comparison/a_full_20class_per_class_accuracy.csv (written by
scripts/analyze_confusable_pair.py) and recomputes the raw per-sample stats
distributions directly from the dataset (cheap, no model needed).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_confusable_pair.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from A.dataset_full import stratified_splits_full

NPZ_PATH = "data/processed/ustc_20class.npz"
RESULTS_DIR = Path("results/comparison")
FIGURES_DIR = Path("results/figures")
HIGHLIGHT = {"Virut", "Neris"}


def plot_per_class_accuracy():
    with (RESULTS_DIR / "a_full_20class_per_class_accuracy.csv").open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["accuracy_mean"]))
    names = [r["class"] for r in rows]
    accs = [float(r["accuracy_mean"]) for r in rows]
    errs = [float(r["accuracy_std"]) for r in rows]
    colors = ["#C44E52" if n in HIGHLIGHT else "#4C72B0" for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, accs, xerr=errs, color=colors, capsize=3)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title("A-Full, 20-class per-class accuracy (3-seed mean)")
    fig.tight_layout()
    out_path = FIGURES_DIR / "a_full_20class_per_class_accuracy.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_pair_distributions():
    dataset, _, _, test_set = stratified_splits_full(NPZ_PATH)
    class_names = dataset.class_names
    stats_np = dataset.stats.numpy()
    labels_np = dataset.labels.numpy()
    test_idx = set(test_set.indices)

    virut_c = class_names.index("Virut")
    neris_c = class_names.index("Neris")
    virut_stats = stats_np[[i for i in test_idx if labels_np[i] == virut_c]]
    neris_stats = stats_np[[i for i in test_idx if labels_np[i] == neris_c]]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, col, label in zip(axes, range(3), ["IAT (ms)", "Jitter (ms)", "Byte entropy"]):
        ax.boxplot([virut_stats[:, col], neris_stats[:, col]], tick_labels=["Virut", "Neris"],
                   showmeans=True)
        ax.set_title(label)
    fig.suptitle("Virut vs Neris: near-identical medians, outlier-driven mean divergence")
    fig.tight_layout()
    out_path = FIGURES_DIR / "confusable_pair_distributions.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_per_class_accuracy()
    plot_pair_distributions()


if __name__ == "__main__":
    main()
