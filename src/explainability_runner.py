import argparse
import logging
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import torch

from src.explainability import debug_misclassifications
from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.models.traffic_clip import TrafficCLIP
from src.utils.utils import get_device, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_gradcam_diagnostic(args, config, target_conflicts):
    config = load_config()
    device = get_device()

    # Paths based on the Training structure
    exp_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}_stats_data{args.use_stats}"

    # model_path = (
    #     Path(__file__).parent.parent
    #     / Path("experiments")
    #     / Path(args.model_version)
    #     / Path(exp_tag)
    # )
    model_path = Path(exp_tag)
    # model_uri = f"models:/{model_path}/latest"
    model_uri = f"models:/best_ht/latest"
    logging.info(f"Attempting to load model from {model_uri}")

    # model_path = model_path / "best_model.pt"
    exp_dir = model_path.parent / Path("gradcam_debug")

    # if not model_path.exists():
    #     logging.error(f"Model checkpoint not found at {model_path}")
    #     return

    # # Initialize model architecture
    # traffic_cfg = config["dataset"]["traffic"]["classes"]
    # num_classes = sum(len(c) for c in traffic_cfg.values())

    # if args.model_version == "optimized":
    #     # use stats flag to toggle statistical features
    #     if args.use_stats:
    #         model = OptimizedTrafficCLIP(
    #             num_classes=num_classes,
    #             use_stats=args.use_stats,
    #             stats_input_dim=args.stats_input_dim,
    #         ).to(device)
    #     else:
    #         model = OptimizedTrafficCLIP(num_classes=num_classes).to(device)
    # else:
    #     model = TrafficCLIP().to(device)

    # if not model_path.exists():
    #     logging.error(f"Weights not found at {model_path}")
    #     return
    try:
        # This loads the entire model object (architecture + weights)
        model = mlflow.pytorch.load_model(model_uri, map_location=device).to(device)
        logging.info(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        logging.error(f"Failed to load model from registry: {e}")

    # # Load the Weights
    # logging.info(f"Loading checkpoint: {model_path}")
    # model.load_state_dict(torch.load(model_path, map_location=device))

    mlflow.set_experiment("TrafficCLIP_GradCAM_Diagnostics")
    with mlflow.start_run(run_name=exp_tag):
        mlflow.log_params(vars(args))
        mlflow.log_param("target_conflicts", str(target_conflicts))

        # Execute Debugging
        figs = debug_misclassifications(
            model, args, config, device, exp_dir, target_conflicts
        )
        logging.info(f"Generated {len(figs)} diagnostic heatmaps for target conflicts.")
        for idx, fig in enumerate(figs):
            mlflow.log_figure(fig, f"diagnostic_plots/heatmap_{idx}.png")

            # plt.close(fig)

        # Log the directory as an artifact
        if exp_dir.exists():
            mlflow.log_artifacts(str(exp_dir), artifact_path="heatmaps_raw")

        logging.info(f"Grad-CAM heatmaps saved to {exp_dir}")


if __name__ == "__main__":
    config = load_config()
    parser = argparse.ArgumentParser(description="TrafficCLIP Grad-CAM Runner")
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
    parser.add_argument(
        "--conflicts",
        nargs="+",
        help="Pairs of True and Predicted labels to debug (e.g., True1 Pred1 True2 Pred2)",
    )
    config = load_config()
    USER_NAME = config["user"]["name"]
    REPO_NAME = config["user"]["repo"]
    dagshub.init(
        repo_owner=USER_NAME,
        repo_name=REPO_NAME,
        mlflow=True,
    )

    import logging

    # from importlib import reload
    # reload(logging)
    # # Updated logging setup to save to a file
    # log_file_name = "explain.log"
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[
    #         logging.StreamHandler(),  # Keep terminal logs
    #         logging.FileHandler(log_file_name),  # Save to this file
    #     ],
    # )
    logger = logging.getLogger("TrafficCLIP")  # Named logger
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Stream (Terminal)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # File
        fh = logging.FileHandler("explain.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    args = parser.parse_args()
    conflict_list = []
    if args.conflicts:
        conflict_list = list(zip(args.conflicts[0::2], args.conflicts[1::2]))
    run_gradcam_diagnostic(args, config, conflict_list)
