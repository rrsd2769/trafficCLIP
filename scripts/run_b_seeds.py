"""
5-seed statistical validation for OptimizedTrafficCLIP: mean and std across
independent trials rather than a single seed=62 run.

The train/val/test split stays fixed at seed=62 for every trial
(src/train_runner.py's SPLIT_SEED, matching A and C), and only args.seed
varies, which controls model init, dropout and shuffle order. So the spread
measures training stochasticity, not five different test sets, and every
trial is evaluated on the same held-out flows as A and C.

Full and V4 only by default: Full is the complete model and V4 the strongest
dynamic-prompt variant. Pass --only to widen or narrow that.

The existing seed=62 row in results/comparison/B_<variant>.csv is reused
rather than re-run, so only 4 new seeds per variant are trained.

Resumable: results/comparison/B_<variant>_seeds.csv accumulates one row per seed;
already-present seeds are skipped unless --force. B_matrix.csv is rebuilt from
whichever *_seeds.csv files exist each time this is run.

Usage:
    HF_HUB_DISABLE_XET=1 MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=.:src \
        .venv/bin/python scripts/run_b_seeds.py
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import mlflow

from src.train_runner import run_experiment
from src.utils.utils import get_device, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/b_seeds.log")],
)
log = logging.getLogger("b_seeds")

# name, model_version, use_stats_prompts, use_stats -- same mapping as
# Mirrors scripts/run_b_matrix.py's VARIANTS.
VARIANT_SPECS = {
    "V1": ("original", False, None),
    "V2": ("original", True, None),
    "V3": ("optimized", False, False),
    "V4": ("optimized", True, False),
    "V5": ("optimized", False, True),
    "Full": ("optimized", True, True),
}

RESULTS_DIR = Path("results/comparison")
LAMBDA_CL = 2.0
DEFAULT_VARIANTS = ["Full", "V4"]
DEFAULT_SEEDS = [62, 63, 64, 65, 66]

ROW_FIELDS = [
    "variant", "model_version", "use_stats_prompts", "use_stats",
    "seed", "accuracy", "precision", "recall", "macro_f1", "seconds",
]
SUMMARY_FIELDS = [
    "variant", "model_version", "use_stats_prompts", "use_stats",
    "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "n_seeds",
]


def read_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def seed_row_from_single_csv(variant):
    """Reuse the existing single-seed=62 B_<variant>.csv row if present."""
    single_path = RESULTS_DIR / f"B_{variant}.csv"
    rows = read_rows(single_path)
    if not rows:
        return None
    r = rows[0]
    if int(r["seed"]) != 62:
        return None
    return {field: r[field] for field in ROW_FIELDS}


def run_one_seed(variant, model_version, use_stats_prompts, use_stats, seed, config, device):
    Args = argparse.Namespace
    run_args = Args(
        lambda_cl=LAMBDA_CL,
        use_stats_prompts=use_stats_prompts,
        model_version=model_version,
        seed=seed,
        use_stats=bool(use_stats),
        stats_input_dim=3,
        lr=1e-4,
        weight_decay=0.01,
    )
    log.info(f"=== {variant} seed={seed}: model_version={model_version} "
              f"use_stats_prompts={use_stats_prompts} use_stats={use_stats} ===")
    t0 = time.time()
    best_f1, best_metrics = run_experiment(run_args, config, device)
    seconds = time.time() - t0
    row = {
        "variant": variant,
        "model_version": model_version,
        "use_stats_prompts": use_stats_prompts,
        "use_stats": use_stats,
        "seed": seed,
        "accuracy": best_metrics.get("accuracy"),
        "precision": best_metrics.get("precision"),
        "recall": best_metrics.get("recall"),
        "macro_f1": best_metrics.get("f1_macro", best_f1),
        "seconds": seconds,
    }
    log.info(f"{variant} seed={seed} acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} ({seconds:.0f}s)")
    return row


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def build_summary():
    summary_rows = []
    for variant in VARIANT_SPECS:
        seeds_path = RESULTS_DIR / f"B_{variant}_seeds.csv"
        rows = read_rows(seeds_path)
        if not rows:
            continue
        acc = [float(r["accuracy"]) for r in rows]
        f1 = [float(r["macro_f1"]) for r in rows]
        acc_mean, acc_std = mean_std(acc)
        f1_mean, f1_std = mean_std(f1)
        summary_rows.append({
            "variant": variant,
            "model_version": rows[0]["model_version"],
            "use_stats_prompts": rows[0]["use_stats_prompts"],
            "use_stats": rows[0]["use_stats"],
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "n_seeds": len(rows),
        })
    summary_path = RESULTS_DIR / "B_matrix.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    log.info(f"Wrote {summary_path} ({len(summary_rows)} variants)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=DEFAULT_VARIANTS,
                         help="Restrict to these variant names.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true",
                         help="Rerun seeds even if already present in B_<variant>_seeds.csv.")
    args_cli = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    config = load_config()
    device = get_device()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    for variant in args_cli.only:
        model_version, use_stats_prompts, use_stats = VARIANT_SPECS[variant]
        seeds_path = RESULTS_DIR / f"B_{variant}_seeds.csv"
        rows = [] if args_cli.force else read_rows(seeds_path)
        done_seeds = {int(r["seed"]) for r in rows}

        # Seed onto the existing single-seed=62 result instead of re-running it.
        if 62 in args_cli.seeds and 62 not in done_seeds:
            reused = seed_row_from_single_csv(variant)
            if reused is not None:
                rows.append(reused)
                done_seeds.add(62)
                log.info(f"{variant}: reused existing seed=62 result from B_{variant}.csv")

        for seed in args_cli.seeds:
            if seed in done_seeds:
                log.info(f"Skipping {variant} seed={seed} -- already present (--force to rerun).")
                continue
            row = run_one_seed(variant, model_version, use_stats_prompts, use_stats, seed, config, device)
            rows.append(row)
            rows.sort(key=lambda r: int(r["seed"]))
            write_rows(seeds_path, rows)  # write after every seed -- resumable

    build_summary()


if __name__ == "__main__":
    main()
