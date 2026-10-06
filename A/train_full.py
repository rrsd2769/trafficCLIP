"""
Trains TrafficCLIPFull (A/model_full.py) and writes results to
results/comparison/. Same protocol as A/train.py (SGD, lr 0.002, cosine
annealing, one-epoch warm-up, batch 64, 200 epochs, 3 seeds averaged).

There is no static text cache here. Without the hypothesis cache, a training
step builds B*K dynamic-prompt strings (this batch's images against all K
class-name hypotheses, each using that image's own stats), tokenizes them in
one call and runs BERT once over the (B*K, L) tensor.

Usage:
    PYTHONPATH=. .venv/bin/python A/train_full.py --npz data/processed/ustc_20class.npz --classes 20
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from A.cache_features import get_device
from A.dataset_full import build_dynamic_prompts, stratified_splits_full
from A.model import BERT_MODEL_NAME
from A.model_full import TrafficCLIPFull
from A.train import build_scheduler, set_seed, supervised_contrastive_loss, write_progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

_FROZEN_PREFIXES = ("semantics_encoder.backbone", "text_encoder")
MAX_LENGTH = 64  # dynamic prompts carry 3 numeric stats on top of the static template; longer than A/cache_features.py's 32.


def trainable_state_dict(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.state_dict().items()
        if not name.startswith(_FROZEN_PREFIXES)
    }


def _hypothesis_forward(model, tokenizer, images, stats, class_names, device, idx=None):
    """
    One hypothesis forward pass. With a hyp-text cache loaded, which is the
    normal case, hypothesis_logits reads the K prompt embeddings from it via
    idx and input_ids/attention_mask go unused. Without one, this tokenizes
    B*K prompts per step, which is far slower.
    """
    input_ids = attention_mask = None
    if idx is None or getattr(model, "_bert_hyp_cache", None) is None:
        prompts = build_dynamic_prompts(class_names, stats)
        encoded = tokenizer(
            prompts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    return model.hypothesis_logits(
        images, stats, input_ids, attention_mask, idx=idx
    )  # ([B, K] logits, [B, 1024] v_e)


@torch.no_grad()
def evaluate_full(model, tokenizer, loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        stats = batch["stats"].to(device)
        logits_hyp, _ = _hypothesis_forward(model, tokenizer, images, stats, class_names, device, idx=batch["idx"])
        all_preds.append(logits_hyp.argmax(dim=-1).cpu())
        all_labels.append(batch["label"])
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def train_one_run(
    npz_path,
    cache_dir,
    hyp_cache_dir,
    num_classes,
    run_seed,
    epochs,
    batch_size,
    base_lr,
    warmup_lr,
    warmup_epochs,
    broken_conv1,
    device,
    log_every,
    checkpoint_path=None,
    progress_path=None,
    run_label="",
    matrix_offset=0,
    matrix_total=1,
):
    set_seed(run_seed)
    dataset, train_set, val_set, test_set = stratified_splits_full(npz_path)
    class_names = dataset.class_names

    model = TrafficCLIPFull(num_classes=num_classes, alpha=0.9, broken_conv1=broken_conv1).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # The stats/text additions do not affect the vision cache, so this is
    # the same f2raw A/cache_features.py builds.
    variant = "broken_conv1" if broken_conv1 else "correct_conv1"
    cache_subdir = Path(cache_dir) / Path(npz_path).stem / variant
    f2raw = np.load(cache_subdir / "resnet_f2raw.npy", mmap_mode="r")
    model.load_vision_cache(f2raw)

    hyp_cache_path = Path(hyp_cache_dir) / Path(npz_path).stem / "bert_pooler_hyp.npy"
    if hyp_cache_path.exists():
        model.load_hyp_text_cache(np.load(hyp_cache_path, mmap_mode="r"))
    else:
        logging.warning(
            f"{hyp_cache_path} not found -- falling back to live per-step BERT calls "
            "(~1000x slower; run A/cache_features_full.py first)."
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=base_lr, momentum=0.9)
    scheduler = build_scheduler(
        optimizer, base_lr, warmup_lr, warmup_epochs, epochs, steps_per_epoch=len(train_loader)
    )

    best_val_f1 = -1.0
    best_state = None
    last_loss = None
    run_t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            stats = batch["stats"].to(device)
            idx = batch["idx"]

            logits_hyp, v_e = _hypothesis_forward(model, tokenizer, images, stats, class_names, device, idx=idx)
            loss_ce = F.cross_entropy(logits_hyp, labels)
            loss_cl = supervised_contrastive_loss(v_e, labels)
            loss = loss_ce + loss_cl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = loss.item()

        val_acc, val_f1 = evaluate_full(model, tokenizer, val_loader, class_names, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = trainable_state_dict(model)

        if epoch % log_every == 0 or epoch == epochs - 1:
            logging.info(
                f"seed={run_seed} epoch={epoch + 1}/{epochs} "
                f"loss={last_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

        if progress_path is not None:
            completed_units = matrix_offset + epoch + 1
            elapsed_this_run = time.time() - run_t0
            sec_per_epoch = elapsed_this_run / (epoch + 1)
            remaining_units = max(0, matrix_total - completed_units)
            write_progress(
                progress_path,
                run_label=run_label,
                seed=run_seed,
                epoch=epoch + 1,
                epochs=epochs,
                run_epoch_units_done=completed_units,
                matrix_total_units=matrix_total,
                overall_percent=round(100 * completed_units / matrix_total, 2),
                val_acc=round(val_acc, 4),
                val_f1=round(val_f1, 4),
                sec_per_epoch=round(sec_per_epoch, 2),
                eta_seconds=round(remaining_units * sec_per_epoch, 1),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trainable_state_dict": best_state or trainable_state_dict(model),
                "alpha": model.alpha,
                "num_classes": num_classes,
                "broken_conv1": broken_conv1,
                "seed": run_seed,
                "npz_path": str(npz_path),
            },
            checkpoint_path,
        )

    test_acc, test_f1 = evaluate_full(model, tokenizer, test_loader, class_names, device)
    return test_acc, test_f1


def main():
    parser = argparse.ArgumentParser(description="Train the tri-modal TrafficCLIPFull (A-Full).")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--cache-dir", default="A/cache")
    parser.add_argument("--hyp-cache-dir", default="A/cache_full")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-lr", type=float, default=0.002)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[62, 63, 64])
    parser.add_argument("--broken-conv1", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint-dir", default="A/checkpoints")
    parser.add_argument(
        "--progress-file",
        default="logs/a_full_progress.json",
        help="Live progress JSON, read by scripts/progress.sh. Set to '' to disable.",
    )
    parser.add_argument("--matrix-offset", type=int, default=0)
    parser.add_argument("--matrix-total-runs", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    logging.info(f"device={device}")

    tag = f"A_Full_{args.classes}class" + ("_brokenconv1" if args.broken_conv1 else "")
    out_path = Path(args.out or f"results/comparison/{tag}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = args.progress_file or None
    matrix_total_runs = args.matrix_total_runs or len(args.seeds)
    matrix_total_units = matrix_total_runs * args.epochs

    rows = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        checkpoint_path = Path(args.checkpoint_dir) / f"{tag}_seed{seed}.pt"
        matrix_offset_units = (args.matrix_offset + i) * args.epochs
        acc, f1 = train_one_run(
            npz_path=args.npz,
            cache_dir=args.cache_dir,
            hyp_cache_dir=args.hyp_cache_dir,
            num_classes=args.classes,
            run_seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            warmup_lr=args.warmup_lr,
            warmup_epochs=args.warmup_epochs,
            broken_conv1=args.broken_conv1,
            device=device,
            log_every=args.log_every,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            run_label=tag,
            matrix_offset=matrix_offset_units,
            matrix_total=matrix_total_units,
        )
        elapsed = time.time() - t0
        logging.info(f"seed={seed} -> test_acc={acc:.4f} test_f1={f1:.4f} ({elapsed:.0f}s)")
        rows.append({"seed": seed, "accuracy": acc, "macro_f1": f1, "seconds": elapsed})

    accs = [r["accuracy"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    total_seconds = sum(r["seconds"] for r in rows)
    rows.append(
        {"seed": "mean", "accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)), "seconds": total_seconds}
    )
    rows.append(
        {"seed": "std", "accuracy": float(np.std(accs)), "macro_f1": float(np.std(f1s)), "seconds": ""}
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "accuracy", "macro_f1", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
