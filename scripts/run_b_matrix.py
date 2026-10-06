"""
Run matrix for OptimizedTrafficCLIP: V1-V5 and Full, matching the
manuscript's Table 1. 10-class, cached, seed 62, one run per variant.

Table 1 mapping, from page 6 of the manuscript:

    Variant | model_version | use_stats_prompts | use_stats
    V1      | original       | False              | --
    V2      | original       | True               | --
    V3      | optimized      | False              | False
    V4      | optimized      | True               | False
    V5      | optimized      | False              | True
    Full    | optimized      | True                | True

Resumable: skips any variant whose results/comparison/B_<variant>.csv
already exists, so a thermal stall or a closed lid costs one variant rather
than the whole matrix.

Usage:
    HF_HUB_DISABLE_XET=1 MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=.:src \
        .venv/bin/python scripts/run_b_matrix.py
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
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/b_matrix.log")],
)
log = logging.getLogger("b_matrix")

VARIANTS = [
    # name, model_version, use_stats_prompts, use_stats
    ("V1", "original", False, None),
    ("V2", "original", True, None),
    ("V3", "optimized", False, False),
    ("V4", "optimized", True, False),
    ("V5", "optimized", False, True),
    ("Full", "optimized", True, True),
]

RESULTS_DIR = Path("results/comparison")
LAMBDA_CL = 2.0
SEED = 62


def write_row(variant, model_version, use_stats_prompts, use_stats, metrics, best_f1, seconds):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"B_{variant}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "model_version",
                "use_stats_prompts",
                "use_stats",
                "seed",
                "accuracy",
                "precision",
                "recall",
                "macro_f1",
                "seconds",
            ]
        )
        writer.writerow(
            [
                variant,
                model_version,
                use_stats_prompts,
                use_stats,
                SEED,
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_macro", best_f1),
                seconds,
            ]
        )
    log.info(f"Wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="*", default=None, help="Restrict to these variant names."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun variants even if their CSV exists."
    )
    args_cli = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    config = load_config()
    device = get_device()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    Args = argparse.Namespace

    for variant, model_version, use_stats_prompts, use_stats in VARIANTS:
        if args_cli.only and variant not in args_cli.only:
            continue

        out_path = RESULTS_DIR / f"B_{variant}.csv"
        if out_path.exists() and not args_cli.force:
            log.info(f"Skipping {variant} -- {out_path} already exists (--force to rerun).")
            continue

        run_args = Args(
            lambda_cl=LAMBDA_CL,
            use_stats_prompts=use_stats_prompts,
            model_version=model_version,
            seed=SEED,
            use_stats=bool(use_stats),
            stats_input_dim=3,
            lr=1e-4,
            weight_decay=0.01,
        )

        log.info(
            f"=== Starting {variant}: model_version={model_version} "
            f"use_stats_prompts={use_stats_prompts} use_stats={use_stats} ==="
        )
        t0 = time.time()
        try:
            best_f1, best_metrics = run_experiment(
                run_args, config, device, is_final=False, is_tune=False
            )
        except Exception:
            log.exception(f"{variant} failed")
            raise
        dt = time.time() - t0

        log.info(
            f"=== {variant} done in {dt:.1f}s: "
            f"acc={best_metrics.get('accuracy')} f1={best_metrics.get('f1_macro')} ==="
        )
        write_row(variant, model_version, use_stats_prompts, use_stats, best_metrics, best_f1, dt)

    log.info("B run matrix complete.")


if __name__ == "__main__":
    main()
