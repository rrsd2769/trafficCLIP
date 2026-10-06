"""Pairwise significance testing across Track B's 6 multi-seed configurations.

Reads results/comparison/B_<variant>_seeds.csv (written by
scripts/run_b_seeds.py) for all of V1, V2, V3, V4, V5, Full, pairs runs by
seed, and runs a paired Wilcoxon signed-rank test on macro_f1 for each of the
15 unordered pairs. Bonferroni-corrects across those 15 comparisons.

The exact two-sided Wilcoxon signed-rank test has a floor of 2 * (1/2)^n on
its p-value for n paired seeds: 0.0625 at n=5, which cannot clear even the
uncorrected 0.05 threshold, and 0.00195 at n=10, which clears the
Bonferroni-corrected 0.05/15 = 0.0033 threshold. The floor for the actual n is
written to the `wilcoxon_p_floor` column. A paired t-test and Cohen's d_z are
reported alongside as a parametric complement with no such floor.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/stats_b_significance.py
"""

import csv
import itertools
from pathlib import Path

import statistics

from scipy.stats import ttest_rel, wilcoxon

RESULTS_DIR = Path("results/comparison")
VARIANTS = ["V1", "V2", "V3", "V4", "V5", "Full"]


def read_seed_series(variant):
    path = RESULTS_DIR / f"B_{variant}_seeds.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {int(r["seed"]): float(r["macro_f1"]) for r in rows}


def main():
    series = {v: read_seed_series(v) for v in VARIANTS}
    missing = [v for v, s in series.items() if s is None]
    if missing:
        raise SystemExit(
            f"Missing results/comparison/B_<variant>_seeds.csv for {missing}; "
            "run scripts/run_b_seeds.py --only V1 V2 V3 V4 V5 Full first."
        )

    pairs = list(itertools.combinations(VARIANTS, 2))
    n_comparisons = len(pairs)
    rows = []
    for a, b in pairs:
        common_seeds = sorted(set(series[a]) & set(series[b]))
        xa = [series[a][s] for s in common_seeds]
        xb = [series[b][s] for s in common_seeds]
        n = len(common_seeds)
        try:
            stat, p = wilcoxon(xa, xb)
        except ValueError as e:
            # all-zero differences (identical macro_f1 across every paired seed)
            stat, p = float("nan"), 1.0
        p_bonf = min(p * n_comparisons, 1.0)
        diffs = [b_ - a_ for a_, b_ in zip(xa, xb)]
        sd = statistics.stdev(diffs)
        d_z = statistics.mean(diffs) / sd if sd > 0 else float("nan")
        t_p = ttest_rel(xa, xb).pvalue
        rows.append({
            "variant_a": a,
            "variant_b": b,
            "n_pairs": n,
            "macro_f1_mean_a": sum(xa) / n,
            "macro_f1_mean_b": sum(xb) / n,
            "wilcoxon_stat": stat,
            "p_value": p,
            "p_bonferroni": p_bonf,
            "significant_bonf_0.05": p_bonf < 0.05,
            "wilcoxon_p_floor": 2 * 0.5 ** n,
            "ttest_p": t_p,
            "ttest_p_bonferroni": min(t_p * n_comparisons, 1.0),
            "cohens_dz": d_z,
        })

    out_path = RESULTS_DIR / "B_significance.csv"
    fields = ["variant_a", "variant_b", "n_pairs", "macro_f1_mean_a", "macro_f1_mean_b",
              "wilcoxon_stat", "p_value", "p_bonferroni", "significant_bonf_0.05",
              "wilcoxon_p_floor", "ttest_p", "ttest_p_bonferroni", "cohens_dz"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_sig = sum(1 for r in rows if r["significant_bonf_0.05"])
    print(f"Wrote {out_path} ({n_comparisons} pairs, {n_sig} significant after Bonferroni)")
    floor = min(r["wilcoxon_p_floor"] for r in rows)
    if floor * n_comparisons >= 0.05:
        print(
            f"\nNote: the Wilcoxon p-value floor at this seed count is {floor:.4f}, "
            f"so no pair can clear Bonferroni at 0.05 regardless of effect size. "
            f"Read the ttest_p and cohens_dz columns instead."
        )

if __name__ == "__main__":
    main()
