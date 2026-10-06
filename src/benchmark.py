import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import (
    get_device,
    get_original_descriptor_bank,
    load_config,
    plot_confusion_matrix,
    save_metrics,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# def test_and_evaluate(
#     model,
#     device,
#     model_type,
#     args,
#     model_version,
#     config,
#     num_runs=3,
#     seed=42,
# ):
#     """
#     Standardized Testing:
#     1. Regenerates a stratified test_loader for each pass using different seeds.
#     2. Averages AC, PR, RC, and Macro F1 across all runs.
#     3. Identifies the highest F1 run for the Confusion Matrix.
#     """
#     model.eval()
#     run_metrics = []
#     best_f1 = -1.0
#     best_preds = None
#     all_labels = None

#     NPZ_PATH = config["paths"]["output_data_file"]
#     TOKENIZER_NAME = config["preprocess"]["tokenizer"]
#     MAX_LENGTH = config["test"]["max_length"]
#     BATCH_SIZE = config["test"]["batch_size"]

#     logging.info(f"Starting evaluation for {model_type} ({num_runs} passes)")

#     for run in range(num_runs):
#         # Generate a unique seed for this specific run
#         current_seed = seed + run
#         set_seed(current_seed)
#         tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

#         # Recreate dataloader with the new seed to get a different stratified test split
#         _, _, test_loader = get_dataloader(
#             npz_path=NPZ_PATH,
#             tokenizer=TOKENIZER_NAME,
#             # prompts=ORIGINAL_PROMPTS,
#             batch_size=BATCH_SIZE,
#             max_length=MAX_LENGTH,
#             seed=current_seed,
#             use_dynamic_prompts=args.use_stats_prompts,  # Phase 2 Toggle
#         )

#         class_names = test_loader.dataset.dataset.class_names

#         preds_list = []
#         labels_list = []

#         # PRE-ENCODE static bank for Original variant
#         if not args.use_stats_prompts:
#             static_descriptor_bank = get_original_descriptor_bank(
#                 model, tokenizer, class_names, MAX_LENGTH, device
#             )

#         with torch.no_grad():
#             for batch in test_loader:
#                 images = batch["image"].to(device)
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["label"].to(device)
#                 stats = batch["stats"].to(device)

#                 # Variant 1: Original Prompts (Global Static Matching)
#                 if not args.use_stats_prompts:
#                     # Vision features normalized for similarity
#                     v_e = model.get_vision_features(images)
#                     # Match against all K classes in the static bank
#                     logits = (
#                         torch.matmul(v_e, static_descriptor_bank.T)
#                         * model.logit_scale.exp()
#                     )
#                     preds = torch.argmax(logits, dim=1)

#                 # Variant 2: Statistical Prompts (Global Dynamic Matching)
#                 else:

#                     v_e = model.get_vision_features(images)
#                     s_e = None
#                     if args.use_stats:
#                         s_e = model.stats_proj(stats)  # [Batch, 512]
#                         s_e = torch.nn.functional.normalize(s_e, p=2, dim=-1)

#                     batch_preds = []

#                     # Must iterate because descriptions depend on specific sample statistics
#                     for i in range(len(images)):
#                         if not args.use_stats_prompts:
#                             # Use the pre-encoded bank you made at the start of the 'run'
#                             t_e_all = static_descriptor_bank  # [K, 1024]
#                         else:
#                             m_iat, m_jitter, m_entropy = stats[i].cpu().numpy()

#                             # Generate K descriptions for THIS specific image's behavior
#                             sample_prompts = [
#                                 f"A network traffic gray photo of class {name} with "
#                                 f"{m_iat:.2f}ms mean IAT, {m_jitter:.2f}ms jitter, "
#                                 f"and {m_entropy:.2f} byte entropy."
#                                 for name in class_names
#                             ]

#                             encoded = tokenizer(
#                                 sample_prompts,
#                                 padding="max_length",
#                                 truncation=True,
#                                 return_tensors="pt",
#                                 max_length=MAX_LENGTH,
#                             ).to(device)
#                         t_e_all = model.get_text_features(
#                             encoded["input_ids"], encoded["attention_mask"]
#                         )  # [K, 1024]

#                         # Expand current image and stats to match K class hypotheses
#                         num_classes = len(class_names)
#                         v_e_hyp = v_e[i : i + 1].expand(num_classes, -1)  # [K, 1024]

#                         if s_e is not None:
#                             s_e_hyp = s_e[i : i + 1].expand(num_classes, -1)  # [K, 512]
#                             combined = torch.cat(
#                                 (v_e_hyp, t_e_all, s_e_hyp), dim=1
#                             )  # [K, 2560]
#                         else:
#                             combined = torch.cat((v_e_hyp, t_e_all), dim=1)  # [K, 2048]

#                         # Pass all K hypotheses through the MLP at once
#                         logits = model.fusion_head(
#                             combined
#                         )  # [K_hypotheses, K_classes]

#                         confidences = torch.diag(logits)
#                         # The predicted class is the one with the highest confidence (diagonal element)
#                         batch_preds.append(torch.argmax(confidences).item())

#                     preds = torch.tensor(batch_preds).to(device)

#             preds_list.extend(preds.cpu().numpy())
#             labels_list.extend(labels.cpu().numpy())

#         # Calculate metrics with zero_division safety
#         acc = accuracy_score(labels_list, preds_list)
#         pr = precision_score(
#             labels_list, preds_list, average="macro", zero_division=0.0
#         )
#         rc = recall_score(labels_list, preds_list, average="macro", zero_division=0.0)
#         f1 = f1_score(labels_list, preds_list, average="macro", zero_division=0.0)

#         # --- MLflow Pass Logging ---
#         mlflow.log_metric("pass_accuracy", acc, step=run)
#         mlflow.log_metric("pass_f1_macro", f1, step=run)
#         mlflow.log_metric("pass_precision", pr, step=run)
#         mlflow.log_metric("pass_recall", rc, step=run)

#         run_metrics.append([acc, pr, rc, f1])

#         if f1 > best_f1:
#             best_f1 = f1
#             best_preds = preds_list
#             all_labels = labels_list
#             # class_names = test_loader.dataset.dataset.class_names

#         logging.info(
#             f"Pass {run+1} (Seed {current_seed}): Accuracy={acc:.4f}, Macro F1={f1:.4f}"
#         )

#     # Average the metrics for the final report
#     avg_metrics = np.mean(run_metrics, axis=0)
#     std_metrics = np.std(run_metrics, axis=0)

#     # Log standard deviation to track model robustness
#     mlflow.log_metric("std_f1_macro", std_metrics[3])

#     logging.info(f"Final Results for {model_type}")
#     logging.info(f"Avg Accuracy (AC):  {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
#     logging.info(f"Avg Precision (PR): {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
#     logging.info(f"Avg Recall (RC):    {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
#     logging.info(f"Avg Macro F1 Score: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")

#     # save run metrics to pandas dataframe
#     # results_path = Path(__file__).parent.parent / "results" / "metrics" / model_version
#     # results_path.mkdir(parents=True, exist_ok=True)
#     # results_file = results_path / f"{model_type}_test_results.csv"
#     # df = pd.DataFrame(
#     #     run_metrics, columns=["Accuracy", "Precision", "Recall", "Macro F1"]
#     # )
#     # df.to_csv(results_file, index_label="Run")
#     # logging.info(f"Saved detailed run metrics to {results_file}")

#     results_path = (
#         Path(__file__).parent.parent / "experiments" / model_version / model_type
#     )
#     results_path.mkdir(parents=True, exist_ok=True)
#     # Log the Confusion Matrix plot
#     fig = plot_confusion_matrix(
#         all_labels, best_preds, class_names, model_type, model_version
#     )
#     fig.savefig(
#         results_path / f"{model_type}_confusion_matrix.png",
#         dpi=100,
#         bbox_inches="tight",
#     )
#     mlflow.log_artifact(str(results_path / f"{model_type}_confusion_matrix.png"))

#     # Log the metrics CSV
#     run_metrics_array = np.array(run_metrics)
#     save_metrics(run_metrics_array, model_type, model_version)
#     metrics_csv = results_path / f"{model_type}_metrics.csv"
#     if metrics_csv.exists():
#         mlflow.log_artifact(str(metrics_csv))

#     return avg_metrics
import copy


def test_and_evaluate(
    model,
    device,
    model_type,
    args,
    model_version,
    config,
    num_runs=3,
    seed=42,
):
    """
    Standardized Testing:
    1. Synchronizes a Master Class List to fix sparse Confusion Matrices.
    2. Regenerates a stratified test_loader for each pass using different seeds.
    3. Averages Metrics and identifies the best run for visualization.
    """
    model.eval()
    run_metrics = []
    best_f1 = -1.0
    best_preds = None
    all_labels = None

    # --- STEP 1: Capture Master Class List ---
    # We do this once to ensure the Confusion Matrix always has the correct dimensions
    NPZ_PATH = config["paths"]["output_data_file"]
    TOKENIZER_NAME = config["preprocess"]["tokenizer"]
    MAX_LENGTH = config["test"]["max_length"]
    BATCH_SIZE = config["test"]["batch_size"]

    # Temporary loader just to get the global taxonomy
    _, _, master_loader = get_dataloader(
        npz_path=NPZ_PATH,
        tokenizer=TOKENIZER_NAME,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    master_class_names = master_loader.dataset.dataset.class_names
    logging.info(
        f"Evaluation started for {model_type}. Global Taxonomy: {len(master_class_names)} classes."
    )

    # Defect 3 fix: the split must be pinned at the seed the model was
    # *trained* on. Regenerating it with `seed + run` per pass drew a fresh
    # stratified split each time, so a model trained on the seed-62 split
    # was then evaluated on splits containing its own training samples --
    # every pass after the first leaked. Since `model` here is a single
    # already-trained checkpoint (not retrained per pass), the training
    # seed's split is the only one this evaluation is valid against; the
    # `num_runs` passes over that fixed split and fixed eval-mode model are
    # deterministic repeats (dropout is off under model.eval()), not an
    # independent resample, so std is honestly ~0. Genuine run-to-run
    # variance requires varying model init/training seed, which is a
    # training-time concern, not something this evaluation function can
    # manufacture after the fact.
    for run in range(num_runs):
        current_seed = seed
        set_seed(current_seed)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        # Recreate dataloader with the fixed training-seed split every pass
        _, _, test_loader = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=current_seed,
            use_dynamic_prompts=args.use_stats_prompts,
        )

        preds_list = []
        labels_list = []

        # PRE-ENCODE static bank for Original variant if needed
        if not args.use_stats_prompts:
            static_descriptor_bank = get_original_descriptor_bank(
                model, tokenizer, master_class_names, MAX_LENGTH, device
            )

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                stats = batch["stats"].to(device)
                idx = batch["idx"]

                # --- Inference Logic ---
                if not args.use_stats_prompts:
                    v_e = model.get_vision_features(images, idx=idx)
                    logits = (
                        torch.matmul(v_e, static_descriptor_bank.T)
                        * model.logit_scale.exp()
                    )
                    preds = torch.argmax(logits, dim=1)
                else:
                    v_e = model.get_vision_features(images, idx=idx)
                    s_e = None
                    if args.use_stats:
                        s_e = torch.nn.functional.normalize(
                            model.stats_proj(stats), p=2, dim=-1
                        )

                    num_classes = len(master_class_names)

                    # Cache fast path: bert_pooler_hyp is keyed by sample idx
                    # and built from the same template as the per-sample
                    # prompts below (verified against
                    # src/cache_features.py's DYNAMIC_TEMPLATE) -- batches
                    # the whole test batch in one cache read instead of one
                    # tokenizer+BERT call per image (this loop was the one
                    # one loop that never got the batched-validation fix
                    # train.py's validate() got; found while wiring the cache
                    # in, and separate from the six named defects).
                    if getattr(model, "cache_loaded", False):
                        t_e_all_batched = model.get_text_features(
                            idx=idx, mode="hyp"
                        )  # [B, K, 1024]
                    else:
                        t_e_all_batched = None

                    batch_preds = []
                    for i in range(len(images)):
                        if t_e_all_batched is not None:
                            t_e_all = t_e_all_batched[i]  # [K, 1024]
                        else:
                            m_iat, m_jitter, m_entropy = stats[i].cpu().numpy()
                            sample_prompts = [
                                f"A network traffic gray photo of class {name} with "
                                f"{m_iat:.2f}ms mean IAT, {m_jitter:.2f}ms jitter, "
                                f"and {m_entropy:.2f} byte entropy."
                                for name in master_class_names
                            ]

                            encoded = tokenizer(
                                sample_prompts,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt",
                                max_length=MAX_LENGTH,
                            ).to(device)

                            t_e_all = model.get_text_features(
                                encoded["input_ids"], encoded["attention_mask"]
                            )
                        v_e_hyp = v_e[i : i + 1].expand(num_classes, -1)

                        if s_e is not None:
                            s_e_hyp = s_e[i : i + 1].expand(num_classes, -1)
                            combined = torch.cat((v_e_hyp, t_e_all, s_e_hyp), dim=1)
                        else:
                            combined = torch.cat((v_e_hyp, t_e_all), dim=1)

                        logits_hyp = model.fusion_head(combined)
                        confidences = torch.diag(logits_hyp)
                        batch_preds.append(torch.argmax(confidences).item())

                    preds = torch.tensor(batch_preds).to(device)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        # Metrics Calculation
        acc = accuracy_score(labels_list, preds_list)
        pr = precision_score(
            labels_list, preds_list, average="macro", zero_division=0.0
        )
        rc = recall_score(labels_list, preds_list, average="macro", zero_division=0.0)
        f1 = f1_score(labels_list, preds_list, average="macro", zero_division=0.0)

        run_metrics.append([acc, pr, rc, f1])

        # Track best run for the Confusion Matrix visualization
        if f1 > best_f1:
            best_f1 = f1
            best_preds = copy.deepcopy(preds_list)
            all_labels = copy.deepcopy(labels_list)

        logging.info(f"Pass {run+1} (Seed {current_seed}): F1={f1:.4f}")

    # --- STEP 2: Aggregated Reporting ---
    avg_metrics = np.mean(run_metrics, axis=0)
    std_metrics = np.std(run_metrics, axis=0)

    logging.info(f"\nFinal Results for {model_type}:")
    logging.info(f"Avg Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    logging.info(f"Avg Macro F1: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")

    # --- STEP 3: Save Artifacts ---
    results_path = (
        Path(__file__).parent.parent / "experiments" / model_version / model_type
    )
    results_path.mkdir(parents=True, exist_ok=True)

    # Plot CM using the MASTER class list and the BEST run data
    fig = plot_confusion_matrix(
        all_labels, best_preds, master_class_names, model_type, model_version
    )
    fig.savefig(
        results_path / f"{model_type}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    mlflow.log_artifact(str(results_path / f"{model_type}_confusion_matrix.png"))
    save_metrics(np.array(run_metrics), model_type, model_version)

    return avg_metrics


if __name__ == "__main__":
    # Load config and set initial environment
    config = load_config()
    device = get_device()

    # Evaluate TrafficClip
    traffic_clip = TrafficCLIP().to(device)
    path_orig = (
        Path(__file__).parent.parent / "saved_models" / "best_traffic_clip_model.pt"
    )
    traffic_clip.load_state_dict(torch.load(path_orig, map_location=device))
    test_seed = config["test"].get("seed", 42)
    test_and_evaluate(
        traffic_clip,
        device,
        "TrafficClip",
        config,
        use_dynamic_prompts=False,
        seed=test_seed,
    )

    # Evaluate TrafficClip Optimized
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(c) for c in traffic_cfg.values())
    opt_model = OptimizedTrafficCLIP(num_classes=num_classes).to(device)
    path_opt = (
        Path(__file__).parent.parent
        / "saved_models"
        / "best_optimized_traffic_clip_model.pt"
    )
    opt_model.load_state_dict(torch.load(path_opt, map_location=device))

    test_and_evaluate(
        opt_model,
        device,
        "TrafficClip_Optimized",
        config,
        use_dynamic_prompts=True,
        seed=test_seed,
    )
