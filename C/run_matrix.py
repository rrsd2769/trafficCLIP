"""
Trains the FlowPic and FS-Net baselines on the existing USTC 10/20-class
data.

Reuses A.dataset.stratified_splits (70/15/15, seed 62), so the test
partition is the same set of flows the other two models are evaluated on.
This holds for --repr original too, since stratified_splits only depends on
the label vector and its order, and C/tests/test_original_repr_alignment.py
gates that the original-representation archives carry the identical y (and
independently, m) as the byte-image archives.

Resumable: each (model, dataset, repr) pair writes its own
results/comparison/C_<model>_<Nclass>[_orig].csv and is skipped if already on
disk unless --force. results/comparison/C_matrix.csv accumulates the mean and
std summary across whichever pairs have been run, merged with earlier runs.

Usage:
    PYTHONPATH=. .venv/bin/python C/run_matrix.py --npz data/processed/ustc_10class.npz --classes 10
    PYTHONPATH=. .venv/bin/python C/run_matrix.py --npz data/processed/ustc_20class.npz --classes 20

    # FlowPic's and FS-Net's own published input representations, once
    # scripts/extract_original_representations.py has been run:
    PYTHONPATH=. .venv/bin/python C/run_matrix.py --classes 10 --repr original
    PYTHONPATH=. .venv/bin/python C/run_matrix.py --classes 20 --repr original
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from A.cache_features import get_device, set_seed
from A.dataset import stratified_splits
from C.baselines.cnn1d import CNN1D
from C.baselines.flowpic import FlowPicCNN
from C.baselines.fsnet import FSNet
from C.baselines.transformer import ByteTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

MODELS = ["flowpic", "fsnet", "cnn1d", "transformer"]
ROW_FIELDS = ["model", "dataset", "seed", "accuracy", "precision", "recall", "macro_f1", "seconds"]
SUMMARY_FIELDS = ["model", "dataset", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "n_seeds"]

# scripts/extract_original_representations.py's output; each model reads a
# different archive under --repr original, since FlowPic's histogram and
# FS-Net's packet-length sequence are different representations of the same
# underlying flows (see C/tests/test_original_repr_alignment.py for the
# alignment guarantee).
ORIGINAL_NPZ = {
    ("flowpic", 10): "data/processed/ustc_10class_flowpic_orig.npz",
    ("flowpic", 20): "data/processed/ustc_20class_flowpic_orig.npz",
    ("fsnet", 10): "data/processed/ustc_10class_fsnet_orig.npz",
    ("fsnet", 20): "data/processed/ustc_20class_fsnet_orig.npz",
}


def build_model(name, num_classes, repr_mode):
    if name == "flowpic":
        return FlowPicCNN(num_classes, img_size=32 if repr_mode == "original" else 28)
    if name == "fsnet":
        if repr_mode == "original":
            return FSNet(num_classes, seq_len=64, input_dim=1)
        return FSNet(num_classes)
    if name == "cnn1d":
        return CNN1D(num_classes)
    if name == "transformer":
        return ByteTransformer(num_classes)
    raise ValueError(f"unknown model {name!r}")


@torch.no_grad()
def evaluate(model, loader, device, model_name):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)[0] if model_name == "fsnet" else model(images)
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(batch["label"])
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return (
        accuracy_score(labels, preds),
        precision_score(labels, preds, average="macro", zero_division=0),
        recall_score(labels, preds, average="macro", zero_division=0),
        f1_score(labels, preds, average="macro", zero_division=0),
    )


def train_one(model_name, npz_path, num_classes, seed, epochs, batch_size, lr, device, repr_mode, recon_weight=0.1):
    set_seed(seed)
    _, train_set, val_set, test_set = stratified_splits(npz_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = build_model(model_name, num_classes, repr_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if model_name == "fsnet":
                logits, recon, seq = model(images)
                loss = F.cross_entropy(logits, labels) + recon_weight * F.mse_loss(recon, seq)
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        _, _, _, val_f1 = evaluate(model, val_loader, device, model_name)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs - 1:
            logging.info(f"{model_name} seed={seed} epoch={epoch + 1}/{epochs} val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    acc, prec, rec, f1 = evaluate(model, test_loader, device, model_name)
    return acc, prec, rec, f1, time.time() - t0


def run_model_dataset(model_name, args, base_tag, out_dir):
    tag_dataset = f"{base_tag}_orig" if args.repr == "original" else base_tag
    out_path = out_dir / f"C_{model_name}_{tag_dataset}.csv"
    if out_path.exists() and not args.force:
        logging.info(f"skip {out_path} (exists, use --force to rerun)")
        with open(out_path) as f:
            return list(csv.DictReader(f))

    if args.repr == "original":
        npz_path = ORIGINAL_NPZ[(model_name, args.classes)]
    else:
        npz_path = args.npz

    rows = []
    for seed in args.seeds:
        acc, prec, rec, f1, seconds = train_one(
            model_name, npz_path, args.classes, seed, args.epochs, args.batch_size, args.lr,
            get_device(), args.repr,
        )
        logging.info(
            f"{model_name} {tag_dataset} seed={seed} acc={acc:.4f} f1={f1:.4f} ({seconds:.0f}s)"
        )
        rows.append(
            {
                "model": model_name,
                "dataset": tag_dataset,
                "seed": seed,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "macro_f1": f1,
                "seconds": seconds,
            }
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"wrote {out_path}")
    return rows


def update_matrix_summary(summary_rows, matrix_path):
    existing = []
    if matrix_path.exists():
        with open(matrix_path) as f:
            existing = list(csv.DictReader(f))
    keys_updated = {(r["model"], r["dataset"]) for r in summary_rows}
    existing = [r for r in existing if (r["model"], r["dataset"]) not in keys_updated]
    merged = existing + summary_rows
    with open(matrix_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    logging.info(f"wrote {matrix_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the FlowPic and FS-Net baselines on the USTC data."
    )
    parser.add_argument("--npz", help="Required for --repr byteimage (default).")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument(
        "--repr", choices=["byteimage", "original"], default="byteimage",
        help="'byteimage' (default): the existing 28x28 flow-byte image, from --npz. "
             "'original': FlowPic's and FS-Net's own published representations, from "
             "ORIGINAL_NPZ (scripts/extract_original_representations.py's output).",
    )
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[62, 63, 64, 65, 66])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="results/comparison")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repr == "byteimage" and not args.npz:
        parser.error("--npz is required for --repr byteimage")

    device = get_device()
    logging.info(f"device={device}")

    tag_dataset = f"{args.classes}class"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name in args.models:
        rows = run_model_dataset(model_name, args, tag_dataset, out_dir)
        accs = [float(r["accuracy"]) for r in rows]
        f1s = [float(r["macro_f1"]) for r in rows]
        summary_rows.append(
            {
                "model": model_name,
                # rows[0]["dataset"], not the outer tag_dataset: run_model_dataset appends
                # "_orig" for --repr original, and the summary key must match that or a
                # second run silently overwrites the byteimage row instead of adding a new one.
                "dataset": rows[0]["dataset"],
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "n_seeds": len(rows),
            }
        )

    update_matrix_summary(summary_rows, Path("results/comparison/C_matrix.csv"))


if __name__ == "__main__":
    main()
