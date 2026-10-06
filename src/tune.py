import argparse
import copy

import mlflow
import optuna
import torch

from src.train_runner import run_experiment
from src.utils.utils import get_device, load_config

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def objective(trial, device, args, config):
    # 1. Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"])
    lambda_cl = trial.suggest_float("lambda_cl", 0.5, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)

    trial_args = copy.deepcopy(args)
    trial_config = copy.deepcopy(config)
    trial_args.lr = lr
    trial_args.lambda_cl = lambda_cl
    trial_args.weight_decay = weight_decay

    trial_config["train"]["optimizer_type"] = optimizer_name
    trial_config["preprocess"]["batch_size"] = batch_size

    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(
            {
                "learning_rate": lr,
                "optimizer": optimizer_name,
                "lambda_cl": lambda_cl,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "trial_number": trial.number,
                "model_version": trial_args.model_version,
            }
        )

        try:
            # run_experiment returns (best_val_f1, best_metrics)
            val_f1, _ = run_experiment(trial_args, trial_config, device, is_tune=True)
            if val_f1 is not None:
                mlflow.log_metric("trial_val_f1", val_f1)
                return val_f1
            return 0.0
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            mlflow.set_tag("status", "failed")
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficCLIP Hyperparameter Tuning")
    parser.add_argument(
        "--n_trials", type=int, default=5, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["original", "optimized"],
        default="optimized",
    )
    # parser.add_argument("--lambda_cl", type=float, default=1.0)
    parser.add_argument(
        "--use_stats", action="store_true", help="Toggle use of statistical features"
    )
    parser.add_argument(
        "--stats_input_dim", type=int, default=3, help="Number of statistical features"
    )
    parser.add_argument(
        "--use_stats_prompts",
        action="store_true",
        help="Toggle dynamic physics prompts",
    )
    parser.add_argument("--seed", type=int, default=62)

    args = parser.parse_args()

    config = load_config()
    device = get_device()

    # Local-only rather than the DagsHub account in config.yaml's user.name;
    # see train_runner.py.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    import logging

    logger = logging.getLogger("TrafficCLIP")  # Named logger
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Stream (Terminal)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # File
        fh = logging.FileHandler("tune.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    mlflow.set_experiment(f"TrafficCLIP_Tuning_{args.model_version}")

    # --- 2. Optuna Optimization Phase ---
    study = optuna.create_study(direction="maximize")

    with mlflow.start_run(run_name=f"Optuna_Parent_{args.model_version}") as parent_run:
        study.optimize(
            lambda t: objective(t, device, args, config), n_trials=args.n_trials
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_overall_f1", study.best_value)

    while mlflow.active_run():
        mlflow.end_run()

    best_lr = study.best_params["lr"]
    logging.info(f"\nOptimization Finished. Best LR: {best_lr:.2e}")

    # --- 3. Final Retraining (Train + Val) ---
    logging.info("Starting Final Retraining on Train + Val combined set...")

    final_args = copy.deepcopy(args)
    final_config = copy.deepcopy(config)

    final_args.lr = study.best_params["lr"]
    final_args.lambda_cl = study.best_params["lambda_cl"]
    final_args.weight_decay = study.best_params["weight_decay"]

    final_config["train"]["optimizer_type"] = study.best_params["optimizer"]
    final_config["preprocess"]["batch_size"] = study.best_params["batch_size"]

    logging.info(
        f"Final Params: LR={final_args.lr:.2e}, "
        f"Lambda={final_args.lambda_cl:.2f}, "
        f"BS={study.best_params['batch_size']}, "
        f"Opt={study.best_params['optimizer']}"
    )

    with mlflow.start_run(run_name=f"Final_Retrain_{args.model_version}"):
        mlflow.log_params(study.best_params)
        mlflow.set_tag("phase", "production_retrain")
        final_f1, final_metrics = run_experiment(
            final_args, final_config, device, is_final=True, is_tune=True
        )

        mlflow.log_metric("final_f1_score", final_f1)
        if final_metrics:
            mlflow.log_metrics(
                {f"final_{k}": v for k, v in final_metrics.items()}
            )
        logging.info(f"Final Retraining Complete. Registered F1: {final_f1:.4f}")
