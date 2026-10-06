"""
Trains TrafficCLIP and writes results to results/comparison/.

Hyperparameters follow the base paper's Sec. 5.1.4: SGD, lr 0.002, cosine
annealing, one-epoch warm-up at 1e-5, batch 64, 200 epochs, three runs
averaged. The paper states neither SGD momentum nor an L_CL/L_CE weighting;
Algorithm 1 sums the losses unweighted, and momentum is set to 0.9 here.

Usage:
    PYTHONPATH=. .venv/bin/python A/train.py --npz data/processed/ustc_20class.npz --classes 20
    PYTHONPATH=. .venv/bin/python A/train.py --npz data/processed/ustc_10class.npz --classes 10
    PYTHONPATH=. .venv/bin/python A/train.py --npz data/processed/ustc_20class.npz --classes 20 --broken-conv1
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

from A.cache_features import get_device
from A.dataset import stratified_splits
from A.model import TrafficCLIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# The frozen backbones are deterministic given (seed, broken_conv1) and are
# never updated, so there is no point writing ~650 MB of ResNet-50/BERT
# weights into every checkpoint.
_FROZEN_PREFIXES = ("semantics_encoder.backbone", "text_encoder")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def write_progress(path, **fields):
    """Write-tmp-then-rename JSON write, so a reader never sees a partial file."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(fields, indent=2))
    os.replace(tmp, path)


def trainable_state_dict(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.state_dict().items()
        if not name.startswith(_FROZEN_PREFIXES)
    }


def supervised_contrastive_loss(v_e, labels, temperature=0.07):
    """
    Eq. 9's L_CL. Positives are same-class samples in the batch, with self
    excluded from both the positive set and the denominator, following
    Khosla et al. (2020); the paper's index bounds leave the self-term open.
    A sample whose class is a singleton in the batch contributes 0 instead
    of dividing by zero, which happens occasionally at 20 classes, batch 64.
    """
    device = v_e.device
    n = v_e.size(0)
    sim = v_e @ v_e.t() / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    labels = labels.to(device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

    log_denom = torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = sim - log_denom

    num_positives = positive_mask.sum(dim=1)
    has_positive = num_positives > 0
    if not has_positive.any():
        return torch.tensor(0.0, device=device)

    # torch.where rather than log_prob * positive_mask: log_prob's diagonal
    # is -inf, and 0 * -inf is NaN, which would poison every row's sum.
    zero = torch.zeros_like(log_prob)
    pos_log_prob_sum = torch.where(positive_mask, log_prob, zero).sum(dim=1)
    per_sample_loss = -pos_log_prob_sum[has_positive] / num_positives[has_positive].clamp(min=1)
    return per_sample_loss.mean()


def build_scheduler(optimizer, base_lr, warmup_lr, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max(1, total_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return warmup_lr / base_lr
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits, _, _, _ = model(images, idx=batch["idx"])
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(batch["label"])
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def train_one_run(
    npz_path,
    cache_dir,
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
    dataset, train_set, val_set, test_set = stratified_splits(npz_path)

    model = TrafficCLIP(alpha=0.9, broken_conv1=broken_conv1).to(device)

    variant = "broken_conv1" if broken_conv1 else "correct_conv1"
    cache_subdir = Path(cache_dir) / Path(npz_path).stem / variant
    f2raw = np.load(cache_subdir / "resnet_f2raw.npy", mmap_mode="r")
    bert_static = np.load(cache_subdir / "bert_pooler_static.npy")
    model.load_vision_cache(f2raw)
    model.load_text_cache(bert_static)

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
            idx = batch["idx"]

            logits, v_e, _, _ = model(images, idx=idx)
            loss_ce = F.cross_entropy(logits, labels)
            loss_cl = supervised_contrastive_loss(v_e, labels)
            loss = loss_ce + loss_cl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)
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
                "broken_conv1": broken_conv1,
                "seed": run_seed,
                "npz_path": str(npz_path),
            },
            checkpoint_path,
        )

    test_acc, test_f1 = evaluate(model, test_loader, device)
    return test_acc, test_f1


def main():
    parser = argparse.ArgumentParser(description="Train TrafficCLIP (A).")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--cache-dir", default="A/cache")
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
        default="logs/a_progress.json",
        help="Live progress JSON, read by scripts/progress.sh. Set to '' to disable.",
    )
    parser.add_argument(
        "--matrix-offset",
        type=int,
        default=0,
        help="Seed-runs already completed elsewhere in a larger matrix.",
    )
    parser.add_argument(
        "--matrix-total-runs",
        type=int,
        default=None,
        help="Total seed-runs across the whole matrix this invocation is part of. Defaults to len(--seeds).",
    )
    args = parser.parse_args()

    device = get_device()
    logging.info(f"device={device}")

    tag = f"A_{args.classes}class" + ("_brokenconv1" if args.broken_conv1 else "")
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
        {
            "seed": "mean",
            "accuracy": float(np.mean(accs)),
            "macro_f1": float(np.mean(f1s)),
            "seconds": total_seconds,
        }
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
