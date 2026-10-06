"""Phase 4: per-class breakdown and confusable-pair analysis for A-Full's
20-class test predictions, across all 3 seeds (correct conv1).

Produces:
  results/comparison/a_full_20class_per_class_accuracy.csv  (task 4.3)
  results/comparison/a_full_20class_confusion_matrix.csv    (summed over 3 seeds)
  results/comparison/confusable_pair_feature_stats.csv      (task 4.1/4.2:
      per-class IAT/jitter/entropy mean/median/std, from the test split's own
      stats array -- these are dataset properties, independent of the model)
  results/figures/a_full_20class_confusion_matrix.png

Reuses the existing test checkpoints under A/checkpoints/A_Full_20class_seed*.pt
(seeds 62, 63, 64) rather than retraining. No training happens here.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/analyze_confusable_pair.py
"""

import csv
import statistics
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from A.cache_features import get_device
from A.dataset_full import stratified_splits_full
from A.model import BERT_MODEL_NAME
from A.model_full import TrafficCLIPFull
from A.train_full import _hypothesis_forward

NPZ_PATH = "data/processed/ustc_20class.npz"
SEEDS = [62, 63, 64]
CHECKPOINT_DIR = Path("A/checkpoints")
CACHE_DIR = Path("A/cache")
HYP_CACHE_DIR = Path("A/cache_full")
RESULTS_DIR = Path("results/comparison")
FIGURES_DIR = Path("results/figures")


@torch.no_grad()
def predict(model, tokenizer, loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        stats = batch["stats"].to(device)
        logits_hyp, _ = _hypothesis_forward(model, tokenizer, images, stats, class_names, device, idx=batch["idx"])
        all_preds.append(logits_hyp.argmax(dim=-1).cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def load_model(seed, device):
    checkpoint_path = CHECKPOINT_DIR / f"A_Full_20class_seed{seed}.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TrafficCLIPFull(num_classes=ckpt["num_classes"], alpha=ckpt["alpha"],
                             broken_conv1=ckpt["broken_conv1"]).to(device)
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)

    variant = "broken_conv1" if ckpt["broken_conv1"] else "correct_conv1"
    cache_subdir = CACHE_DIR / Path(NPZ_PATH).stem / variant
    f2raw = np.load(cache_subdir / "resnet_f2raw.npy", mmap_mode="r")
    model.load_vision_cache(f2raw)

    hyp_cache_path = HYP_CACHE_DIR / Path(NPZ_PATH).stem / "bert_pooler_hyp.npy"
    model.load_hyp_text_cache(np.load(hyp_cache_path, mmap_mode="r"))
    return model


def main():
    device = get_device()
    dataset, _, _, test_set = stratified_splits_full(NPZ_PATH)
    class_names = dataset.class_names
    n_classes = len(class_names)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    summed_cm = np.zeros((n_classes, n_classes), dtype=int)
    per_seed_class_acc = {c: [] for c in class_names}

    for seed in SEEDS:
        model = load_model(seed, device)
        labels, preds = predict(model, tokenizer, test_loader, class_names, device)
        cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
        summed_cm += cm
        for i, name in enumerate(class_names):
            row_sum = cm[i].sum()
            per_seed_class_acc[name].append(cm[i, i] / row_sum if row_sum else float("nan"))
        print(f"seed={seed}: overall accuracy={np.trace(cm) / cm.sum():.4f}")

    # Task 4.3: per-class accuracy table, averaged over 3 seeds.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_class_path = RESULTS_DIR / "a_full_20class_per_class_accuracy.csv"
    with per_class_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy_mean", "accuracy_std", "n_test_samples"])
        n_test_per_class = len(test_set) // n_classes  # stratified, so roughly balanced
        for name in class_names:
            accs = per_seed_class_acc[name]
            writer.writerow([name, statistics.mean(accs), statistics.pstdev(accs), n_test_per_class])
    print(f"Wrote {per_class_path}")

    cm_path = RESULTS_DIR / "a_full_20class_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for i, name in enumerate(class_names):
            writer.writerow([name] + summed_cm[i].tolist())
    print(f"Wrote {cm_path}")

    # Find the largest off-diagonal confusion pair (summed both directions).
    off_diag = summed_cm.copy()
    np.fill_diagonal(off_diag, 0)
    symmetric_confusion = off_diag + off_diag.T
    i, j = np.unravel_index(np.argmax(symmetric_confusion), symmetric_confusion.shape)
    pair_errors = int(off_diag[i, j] + off_diag[j, i])
    total_errors = int(off_diag.sum())
    print(f"Top confusable pair: {class_names[i]} <-> {class_names[j]}, "
          f"{pair_errors}/{total_errors} errors ({100 * pair_errors / total_errors:.1f}%)")

    # Task 4.1/4.2: per-class stats (IAT, jitter, entropy) from the test split itself.
    stats_np = dataset.stats.numpy()
    labels_np = dataset.labels.numpy()
    test_indices = test_set.indices
    stats_path = RESULTS_DIR / "confusable_pair_feature_stats.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "n", "iat_mean", "iat_median", "iat_std",
                          "jitter_mean", "jitter_median", "jitter_std",
                          "entropy_mean", "entropy_median", "entropy_std"])
        for c, name in enumerate(class_names):
            class_test_idx = [idx for idx in test_indices if labels_np[idx] == c]
            class_stats = stats_np[class_test_idx]  # [n, 3] = iat, jitter, entropy
            row = [name, len(class_test_idx)]
            for col in range(3):
                vals = class_stats[:, col]
                row.extend([float(np.mean(vals)), float(np.median(vals)), float(np.std(vals))])
            writer.writerow(row)
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
