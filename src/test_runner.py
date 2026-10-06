import argparse
import logging
from pathlib import Path

import dagshub
import mlflow
import torch

from src.benchmark import test_and_evaluate
from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.models.traffic_clip import TrafficCLIP
from src.utils.utils import get_device, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_test_experiment(args):
    config = load_config()
    device = get_device()

    # Paths based on the Training structure
    if args.use_stats:
        exp_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}_stats_data{args.use_stats}"
    else:
        exp_tag = (
            f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}"
        )
    # if args.use_stats_prompts:
    #     exp_tag += "_statsTrue"
    # else:
    #     exp_tag += "_statsFalse"

    mlflow.set_experiment("TrafficCLIP_Test_Evaluation_1")
    # results_path = (
    #     Path(__file__).parent.parent
    #     / Path("experiments")
    #     / Path(args.model_version)
    #     / Path(exp_tag)
    # )
    model_path = Path(exp_tag)
    # model_uri = f"models:/{model_path}/latest"
    model_uri = f"models:/best_ht/latest"
    logging.info(f"Attempting to load model from {model_uri}")

    # if not model_path.exists():
    #     logging.error(f"Model checkpoint not found at {model_path}")
    #     return

    try:
        # This loads the entire model object (architecture + weights)
        model = mlflow.pytorch.load_model(model_uri, map_location=device).to(device)
        logging.info(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        logging.error(f"Failed to load model from registry: {e}")

    # # Initialize model architecture
    # traffic_cfg = config["dataset"]["traffic"]["classes"]
    # num_classes = sum(len(c) for c in traffic_cfg.values())

    # if args.model_version == "optimized":
    #     if args.use_stats:
    #         model = OptimizedTrafficCLIP(
    #             num_classes=num_classes,
    #             use_stats=True,
    #             stats_input_dim=args.stats_input_dim,
    #         ).to(device)
    #     else:
    #         model = OptimizedTrafficCLIP(num_classes=num_classes, use_stats=False).to(
    #             device
    #         )
    # else:
    #     model = TrafficCLIP().to(device)

    # # Load the Weights
    # logging.info(f"Loading checkpoint: {model_path}")
    # model.load_state_dict(torch.load(model_path, map_location=device))

    # --- MLflow Evaluation Run ---
    with mlflow.start_run(run_name=f"Test_{exp_tag}"):
        mlflow.log_params(vars(args))  # Log test-time parameters
        avg_results = test_and_evaluate(
            model=model,
            device=device,
            model_version=args.model_version,
            model_type=exp_tag,
            args=args,
            config=config,
            num_runs=args.num_runs,
            seed=args.seed,
        )

        # Log Summary Averages
        mlflow.log_metrics(
            {
                "avg_accuracy": avg_results[0],
                "avg_precision": avg_results[1],
                "avg_recall": avg_results[2],
                "avg_f1_macro": avg_results[3],
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficCLIP Ablation Test Runner")
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--lambda_cl", type=float, required=True)
    parser.add_argument("--use_stats_prompts", action="store_true")
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["original", "optimized"],
        default="optimized",
    )
    parser.add_argument(
        "--use_stats", action="store_true", help="Toggle use of statistical features"
    )
    parser.add_argument(
        "--stats_input_dim", type=int, default=3, help="Number of statistical features"
    )
    parser.add_argument("--seed", type=int, default=42)
    config = load_config()
    USER_NAME = config["user"]["name"]
    REPO_NAME = config["user"]["ht_repo"]
    dagshub.init(
        repo_owner=USER_NAME,
        repo_name=REPO_NAME,
        mlflow=True,
    )

    import logging

    # from importlib import reload
    # reload(logging)

    logger = logging.getLogger("TrafficCLIP")  # Named logger
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Stream (Terminal)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # File
        fh = logging.FileHandler("testing.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    args = parser.parse_args()
    run_test_experiment(args)
