"""Bar chart of Track B's 6 configurations (accuracy and macro-F1) with 95%
confidence-interval error bars, from the multi-seed results written by
scripts/run_b_seeds.py.

95% CI is computed as mean +/- 1.96 * std / sqrt(n), the normal approximation;
with n=5 seeds this is wide, which the chart shows honestly rather than
hiding behind a bare mean.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_b_variance.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/comparison")
FIGURES_DIR = Path("results/figures")
VARIANTS = ["V1", "V2", "V3", "V4", "V5", "Full"]
Z_95 = 1.96


def read_seed_series(variant, field):
    path = RESULTS_DIR / f"B_{variant}_seeds.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [float(r[field]) for r in rows]


def mean_ci95(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = var ** 0.5
    return mean, Z_95 * std / (n ** 0.5)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for field, ylabel, fname in [
        ("accuracy", "Accuracy", "b_multiseed_accuracy.png"),
        ("macro_f1", "Macro F1", "b_multiseed_macro_f1.png"),
    ]:
        means, errs = [], []
        for v in VARIANTS:
            m, e = mean_ci95(read_seed_series(v, field))
            means.append(m)
            errs.append(e)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(VARIANTS, means, yerr=errs, capsize=4, color="#4C72B0")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Track B, 5-seed {ylabel} (95% CI)")
        fig.tight_layout()
        out_path = FIGURES_DIR / fname
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
