import logging
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from early_stopping import EarlyStopping
from loss import contrastive_loss_func
from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import (
    get_device,
    get_original_descriptor_bank,
    load_config,
    plot_convergence,
    save_metrics,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def validate(
    model,
    model_version,
    args,
    val_loader,
    device,
    config,
    lambda_cl=5.0,
    class_names=None,
):
    """
    Standardized Validation Function:
    Calculates performance metrics using joint loss (CE + CL).
    """
    model.eval()
    all_preds = []  # Total predictions across all batches
    all_labels = []  # Total true labels across all batches
    total_val_loss = 0.0

    criterion_ce = torch.nn.CrossEntropyLoss()

    # Load configuration
    NPZ_PATH = config["paths"]["output_data_file"]
    TOKENIZER_NAME = config["preprocess"]["tokenizer"]
    MAX_LENGTH = config["test"]["max_length"]
    BATCH_SIZE = config["test"]["batch_size"]

    # class_names = val_loader.dataset.dataset.class_names

    class_names_list = (
        class_names
        if class_names is not None
        else val_loader.dataset.dataset.class_names
    )
    class_prompts = [
        f"A network traffic gray photo of class {name}." for name in class_names_list
    ]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    class_tokens = tokenizer(
        class_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=config["preprocess"]["max_length"],
    ).to(device)

    # val_loader = get_dataloader(
    #     npz_path=NPZ_PATH,
    #     tokenizer=TOKENIZER_NAME,
    #     batch_size=BATCH_SIZE,
    #     max_length=MAX_LENGTH,
    #     use_dynamic_prompts=use_dynamic_prompts,
    # )

    num_classes = len(class_names_list)

    # Batched-validation fix: builds all B*K hypothesis prompts for a batch
    # in one list comprehension, tokenizes them in a single call, and runs
    # BERT once over the whole (B*K, ...) tensor -- replacing what used to
    # be a per-sample Python loop with its own tokenizer + BERT call per
    # image (B calls of K prompts each, e.g. 64 x 10 = 640 tiny forward
    # passes per batch). Identical math, ~10-50x fewer, much larger calls.
    def _hypothesis_text_features(stats_batch, idx=None):
        # Cache fast path: bert_pooler_hyp is keyed by sample idx and built
        # from the exact same template as the prompts below (verified against
        # src/cache_features.py's DYNAMIC_TEMPLATE), so when a cache is loaded
        # this reproduces the tokenizer/BERT path exactly, skipping both.
        if idx is not None and getattr(model, "cache_loaded", False):
            return model.get_text_features(idx=idx, mode="hyp")  # [B, K, 1024]

        b = stats_batch.size(0)
        stats_np = stats_batch.detach().cpu().numpy()
        prompts = [
            f"A network traffic gray photo of class {name} with "
            f"{stats_np[i, 0]:.2f}ms mean IAT, {stats_np[i, 1]:.2f}ms jitter, "
            f"and {stats_np[i, 2]:.2f} byte entropy."
            for i in range(b)
            for name in class_names_list
        ]
        encoded = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH,
        ).to(device)
        t_e_flat = model.get_text_features(
            encoded["input_ids"], encoded["attention_mask"]
        )  # [B*K, 1024]
        return t_e_flat.view(b, num_classes, -1)  # [B, K, 1024]

    # PRE-ENCODE static bank for Original variant
    if not args.use_stats_prompts:
        static_descriptor_bank = get_original_descriptor_bank(
            model, tokenizer, class_names_list, MAX_LENGTH, device
        )

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            stats = batch["stats"].to(device)
            idx = batch["idx"]
            batch_size = images.size(0)

            # LOSS CALCULATION
            if model_version == "original":
                logits_gt, current_scale = model(
                    images, class_tokens.input_ids, class_tokens.attention_mask, idx=idx
                )
            else:
                logits_gt, current_scale = model(
                    images, input_ids, attention_mask, stats, idx=idx
                )

            loss_ce = criterion_ce(logits_gt, labels)
            v_f = model.get_vision_features(images, idx=idx)
            loss_cl = contrastive_loss_func(v_f, labels, current_scale)

            loss = loss_ce + (lambda_cl * loss_cl)
            total_val_loss += loss.item()

            # PREDICTION STEP
            v_e = model.get_vision_features(images, idx=idx)  # [B, 1024]

            # PATH A: Original Model (Cosine Similarity)
            if model_version == "original":
                if not args.use_stats_prompts:
                    # Static Matching
                    logits = (
                        torch.matmul(v_e, static_descriptor_bank.T)
                        * model.logit_scale.exp()
                    )
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                else:
                    # Dynamic Matching for Original -- batched over B*K
                    t_e_all = _hypothesis_text_features(stats, idx=idx)  # [B, K, 1024]
                    sim = (
                        torch.einsum("bd,bkd->bk", v_e, t_e_all)
                        * model.logit_scale.exp()
                    )
                    batch_preds = torch.argmax(sim, dim=1).cpu().numpy().tolist()

            # PATH B: Optimized Model (MLP Fusion)
            else:
                s_e = None
                if args.use_stats:
                    s_e = model.stats_proj(stats)  # [B, 512]
                    s_e = torch.nn.functional.normalize(s_e, p=2, dim=-1)

                if not args.use_stats_prompts:
                    # Broadcast the shared static bank across the batch.
                    t_e_all = static_descriptor_bank.unsqueeze(0).expand(
                        batch_size, -1, -1
                    )  # [B, K, 1024]
                else:
                    t_e_all = _hypothesis_text_features(stats, idx=idx)  # [B, K, 1024]

                v_e_hyp = v_e.unsqueeze(1).expand(-1, num_classes, -1)  # [B, K, 1024]

                if s_e is not None:
                    s_e_hyp = s_e.unsqueeze(1).expand(-1, num_classes, -1)  # [B, K, 512]
                    combined = torch.cat(
                        (v_e_hyp, t_e_all, s_e_hyp), dim=-1
                    )  # [B, K, 2560]
                else:
                    combined = torch.cat((v_e_hyp, t_e_all), dim=-1)  # [B, K, 2048]

                # Pass all B*K hypotheses through the MLP in one call --
                # nn.Linear/LayerNorm/ReLU/Dropout all operate on the last
                # dimension, so a [B, K, D] input needs no reshaping.
                logits_hyp = model.fusion_head(combined)  # [B, K, K_classes]
                # The predicted class per sample is the hypothesis whose own
                # class-index logit is highest -- the batched equivalent of
                # torch.diag() per sample.
                confidences = torch.diagonal(logits_hyp, dim1=1, dim2=2)  # [B, K]
                batch_preds = torch.argmax(confidences, dim=1).cpu().numpy().tolist()

            all_preds.extend(batch_preds)
            all_labels.extend(labels.cpu().numpy())

    # Standardized Performance Metrics
    metrics = {
        "loss": total_val_loss / len(val_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    return metrics


# def train(
#     model,
#     model_version,
#     model_type,
#     train_loader,
#     val_loader,
#     config,
#     device,
#     lambda_cl,
#     early_stopping=None,
# ):
#     """
#     Standardized Training Loop:
#     Implements joint optimization using Cross-Entropy and Contrastive Loss.
#     """

#     epochs = config["train"][model_version]["epochs"]
#     optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#     criterion_ce = nn.CrossEntropyLoss()

#     # History dictionary for convergence plot
#     history = {"train_loss": [], "val_loss": [], "val_f1": []}
#     best_f1 = 0.0
#     model_path = (
#         Path(__file__).parent.parent
#         / Path("experiments")
#         / Path(model_version)
#         # / f"{model_type}_L{lambda_cl}"
#         / model_type
#     )
#     model_path.mkdir(parents=True, exist_ok=True)

#     run_metrics = []
#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         total_train_loss = 0.0

#         # Warm-up strategy for epoch 1
#         if epoch == 0:
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = 1e-5

#         for batch in train_loader:
#             images = batch["image"].to(device)
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["label"].to(device)
#             # Extract Physics Modality (IAT, Jitter, Entropy)
#             stats_vector = batch["stats"].to(device)

#             optimizer.zero_grad()
#             if model_version == "original":
#                 logits, current_scale = model(images, input_ids, attention_mask)
#             else:
#                 logits, current_scale = model(
#                     images, input_ids, attention_mask, stats_vector
#                 )

#             # Joint optimization: CE + CL
#             loss_ce = criterion_ce(logits, labels)
#             v_f = model.get_vision_features(images)
#             loss_cl = contrastive_loss_func(v_f, labels, current_scale)

#             loss = loss_ce + (lambda_cl * loss_cl)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         if epoch > 0:
#             scheduler.step()

#         # Validation Phase
#         val_metrics = validate(model, model_version, val_loader, device, lambda_cl)

#         val_loss = val_metrics["loss"]
#         val_acc = val_metrics["accuracy"]
#         val_pre = val_metrics["precision"]
#         val_re = val_metrics["recall"]
#         val_f1 = val_metrics["f1_macro"]

#         history["train_loss"].append(total_train_loss / len(train_loader))
#         history["val_loss"].append(val_loss)
#         history["val_f1"].append(val_f1)

#         logging.info(
#             f"Epoch {epoch+1}/{epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | "
#             f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
#         )

#         # Check early stopping condition
#         early_stopping(val_f1)
#         if early_stopping.stop_training:
#             logging.info(
#                 f"Early stopping triggered at epoch {epoch+1}. Training terminated."
#             )
#             break

#         # Save the best model based on Macro F1 score
#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             logging.info(model_path)
#             torch.save(model.state_dict(), model_path / "best_model.pt")
#             logging.info(f"Best model saved with F1: {val_f1:.4f}")

#     run_metrics.append([val_acc, val_pre, val_re, val_f1])
#     save_metrics(run_metrics, model_type, model_version)
#     plot_convergence(history, model_type, save_path=model_path)


def train(
    model,
    model_version,
    model_type,
    train_loader,
    val_loader,
    args,
    config,
    device,
    lambda_cl,
    early_stopping=None,
    optimizer=None,
    scheduler=None,
    is_tune=False,
    is_final=False,
    class_names=None,
):
    """
    Optimized Tri-modal Training Loop
    """

    if is_tune and not is_final:
        epochs = config["train"]["tuning_epochs"]
    else:
        epochs = config["train"][model_version]["epochs"]
    warmup_epochs = 5
    lr = args.lr if hasattr(args, "lr") else 1e-4

    if optimizer is None:
        wd = args.weight_decay if hasattr(args, "weight_decay") else 0.01
        opt_type = config["train"].get("optimizer_type", "adamw").lower()

        if opt_type == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
            )
        else:  # Default to Adam
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if scheduler is None:
        sched_type = config["train"].get("scheduler_type", "cosine").lower()
        if sched_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif sched_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3
            )
        else:
            scheduler = None

    criterion_ce = nn.CrossEntropyLoss()
    TOKENIZER_NAME = config["preprocess"]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # class_prompts = [
    #     f"A network traffic gray photo of class {name}."
    #     for name in train_loader.dataset.dataset.class_names
    # ]
    # class_tokens = tokenizer(
    #     class_prompts,
    #     padding=True,
    #     truncation=True,
    #     return_tensors="pt",
    #     max_length=config["preprocess"]["max_length"],
    # ).to(device)
    # logging.info("Number of classes: %d", len(train_loader.dataset.dataset.class_names))

    class_names_list = (
        class_names
        if class_names is not None
        else train_loader.dataset.dataset.class_names
    )
    class_prompts = [
        f"A network traffic gray photo of class {name}." for name in class_names_list
    ]
    # logging.info(
    #     "Master Class List: %d classes", len(train_loader.dataset.dataset.class_names)
    # )

    # logging.info("Number of classes: %d", len(val_loader.dataset.dataset.class_names))
    # actual_labels = train_loader.dataset.dataset.labels
    # num_classes = max(actual_labels) + 1
    # logging.info("Max label index: %d", num_classes - 1)
    class_tokens = tokenizer(
        class_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=config["preprocess"]["max_length"],
    ).to(device)
    # History dictionary for convergence plot
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0
    best_metrics = None
    model_path = (
        Path(__file__).parent.parent
        / Path("experiments")
        / Path(model_version)
        # / f"{model_type}_L{lambda_cl}"
        / model_type
    )
    model_path.mkdir(parents=True, exist_ok=True)

    run_metrics = []
    for epoch in range(epochs):
        # Training Phase
        model.train()
        # Defect 6 fix: model.train() alone leaves the frozen submodules'
        # `.training` flag flipped too -- requires_grad=False freezes
        # parameters, not behaviour. Left unpinned, BERT's dropout stays
        # active in every live forward pass (a hard crash on MPS, since
        # scaled_dot_product_attention there doesn't support dropout) and
        # ResNet-50's BatchNorm running stats keep drifting despite the
        # backbone being frozen. See docs/feature_cache.md.
        model.semantics_backbone.eval()
        model.text_encoder.eval()
        total_train_loss = 0.0

        # Warm-up strategy for epoch 1
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-5

        for batch in train_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # Extract Physics Modality (IAT, Jitter, Entropy)
            stats_vector = batch["stats"].to(device)
            idx = batch["idx"]

            optimizer.zero_grad()
            if model_version == "original":
                logits, current_scale = model(
                    images,
                    class_tokens.input_ids,
                    class_tokens.attention_mask,
                    idx=idx,
                )
            else:
                logits, current_scale = model(
                    images, input_ids, attention_mask, stats_vector, idx=idx
                )

            # Joint optimization: CE + CL
            loss_ce = criterion_ce(logits, labels)
            v_f = model.get_vision_features(images, idx=idx)
            loss_cl = contrastive_loss_func(v_f, labels, current_scale)

            # loss = loss_ce + (current_lambda * loss_cl)
            loss = loss_ce + (lambda_cl * loss_cl)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        if epoch > 0:
            scheduler.step()

        # Validation Phase
        if model_version == "original":
            val_metrics = validate(
                model,
                model_version,
                device=device,
                args=args,
                val_loader=val_loader,
                config=config,
                lambda_cl=lambda_cl,
            )
        else:
            if not is_final:
                val_metrics = validate(
                    model,
                    model_version,
                    args=args,
                    device=device,
                    val_loader=val_loader,
                    config=config,
                    lambda_cl=lambda_cl,
                )
            else:
                val_metrics = validate(
                    model,
                    model_version,
                    args=args,
                    device=device,
                    val_loader=val_loader,
                    config=config,
                    lambda_cl=lambda_cl,
                    class_names=class_names,
                )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_pre = val_metrics["precision"]
        val_re = val_metrics["recall"]
        val_f1 = val_metrics["f1_macro"]

        history["train_loss"].append(total_train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # --- MLflow Metric Logging ---
        mlflow.log_metric(
            "train_loss", total_train_loss / len(train_loader), step=epoch
        )
        mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
        mlflow.log_metric("val_acc", val_metrics["accuracy"], step=epoch)
        mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
        mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)
        mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"], step=epoch)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Check early stopping condition
        early_stopping(val_f1)
        if early_stopping.stop_training:
            logging.info(
                f"Early stopping triggered at epoch {epoch+1}. Training terminated."
            )
            break

        # Save the best model based on Macro F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = dict(val_metrics)
            logging.info(model_path)
            torch.save(model.state_dict(), model_path / "best_model.pt")
            # mlflow.pytorch.log_model(model, name="best_model")
            logging.info(f"Best model saved with F1: {val_f1:.4f}")

    # Finalize: Load best model, log to MLflow, save metrics, and plot convergence
    best_weights = torch.load(model_path / "best_model.pt", map_location=device)
    model.load_state_dict(best_weights)

    # MLflow 3.x defaults serialization_format to "pt2" (torch.export trace),
    # which raises unless input_example is given (it virtually executes
    # forward() to trace the graph). "pickle" is the pre-3.x behaviour --
    # a plain pickle of the module -- and needs no example input.
    mlflow.pytorch.log_model(model, name="best_model", serialization_format="pickle")

    run_metrics.append([val_acc, val_pre, val_re, val_f1])
    save_metrics(run_metrics, model_type, model_version)
    fig = plot_convergence(history, model_type, save_path=model_path)
    mlflow.log_figure(fig, f"{model_type}_convergence.png")
    mlflow.log_artifact(str(model_path / f"{model_type}_convergence.png"))

    return best_f1, (best_metrics or {})


if __name__ == "__main__":

    # load config
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # set device
    device = get_device()

    try:
        NPZ_PATH = config["paths"]["output_data_file"]
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = config["train"]["seed"]
        BATCH_SIZE = config["preprocess"]["batch_size"]
        LAMBDA_CL_ORIGINAL = config["train"]["traffic_clip"]["lambda_cl"]
        LAMBDA_CL_OPTIMIZED = config["train"]["optimized_traffic_clip"]["lambda_cl"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # set seed for reproducibility
    set_seed(SEED)

    # create dataloaders for TrafficClip
    try:
        train_loader_original, val_loader_original, _ = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            # prompts=ORIGINAL_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
            use_dynamic_prompts=False,
        )

        logging.info("DataLoaders created successfully for TrafficClip.")
    except Exception as e:
        logging.error(f"Error creating DataLoaders for TrafficClip: {e}")
        raise

    # initialize original trafficclip model
    traffic_clip = TrafficCLIP()
    traffic_clip.to(device)

    # train original traffic clip model
    patience_traffic_clip = config["early_stopping"]["traffic_clip"]["patience"]
    delta_traffic_clip = config["early_stopping"]["traffic_clip"]["delta"]
    early_stopping_traffic_clip = EarlyStopping(
        patience=patience_traffic_clip,
        delta=delta_traffic_clip,
        verbose=True,
        mode="max",
    )
    logging.info("Starting training for TrafficCLIP")
    try:
        train(
            model=traffic_clip,
            model_type="traffic_clip",
            train_loader=train_loader_original,
            val_loader=val_loader_original,
            config=config,
            device=device,
            lambda_cl=LAMBDA_CL_ORIGINAL,
            early_stopping=early_stopping_traffic_clip,
        )
    except Exception as e:
        logging.error(f"Error during training TrafficCLIP: {e}")
        raise

    # create dataloaders for TrafficClip Optimized
    try:
        train_loader_optimized, val_loader_optimized, _ = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            # prompts=SEMANTIC_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
            use_dynamic_prompts=True,
        )

        logging.info("DataLoaders created successfully for TrafficClip Optimized.")
    except Exception as e:
        logging.error(f"Error creating DataLoaders for TrafficClip Optimized: {e}")
        raise

    # initialize optimized trafficclip model
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
    optimized_traffic_clip = OptimizedTrafficCLIP(num_classes=num_classes)
    optimized_traffic_clip.to(device)

    patience_traffic_clip_optimized = config["early_stopping"][
        "optimized_traffic_clip"
    ]["patience"]
    delta_traffic_clip_optimized = config["early_stopping"]["optimized_traffic_clip"][
        "delta"
    ]

    # train optimized traffic clip model
    early_stopping_optimized_traffic_clip = EarlyStopping(
        patience=patience_traffic_clip_optimized,
        delta=delta_traffic_clip_optimized,
        verbose=True,
        mode="max",
    )
    logging.info("Starting training for OptimizedTrafficCLIP")
    try:
        train(
            model=optimized_traffic_clip,
            model_type="optimized_traffic_clip",
            train_loader=train_loader_optimized,
            val_loader=val_loader_optimized,
            config=config,
            device=device,
            lambda_cl=LAMBDA_CL_OPTIMIZED,
            early_stopping=early_stopping_optimized_traffic_clip,
        )
    except Exception as e:
        logging.error(f"Error during training OptimizedTrafficCLIP: {e}")
        raise
