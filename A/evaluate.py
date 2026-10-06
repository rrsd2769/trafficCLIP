"""
Re-evaluates a checkpoint written by A/train.py against the same fixed test
split, without re-running training.

Usage:
    PYTHONPATH=. .venv/bin/python A/evaluate.py \\
        --checkpoint A/checkpoints/A_20class_seed62.pt \\
        --npz data/processed/ustc_20class.npz
"""

import argparse
import logging

import numpy as np
import torch

from A.cache_features import get_device
from A.dataset import stratified_splits
from A.model import TrafficCLIP
from A.train import evaluate
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved A checkpoint on the test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--cache-dir", default="A/cache")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    logging.info(
        f"Loaded checkpoint: seed={ckpt['seed']} alpha={ckpt['alpha']} "
        f"broken_conv1={ckpt['broken_conv1']} npz={ckpt['npz_path']}"
    )

    model = TrafficCLIP(alpha=ckpt["alpha"], broken_conv1=ckpt["broken_conv1"]).to(device)
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)

    variant = "broken_conv1" if ckpt["broken_conv1"] else "correct_conv1"
    from pathlib import Path

    cache_subdir = Path(args.cache_dir) / Path(args.npz).stem / variant
    f2raw = np.load(cache_subdir / "resnet_f2raw.npy", mmap_mode="r")
    bert_static = np.load(cache_subdir / "bert_pooler_static.npy")
    model.load_vision_cache(f2raw)
    model.load_text_cache(bert_static)

    _, _, _, test_set = stratified_splits(args.npz)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    acc, f1 = evaluate(model, test_loader, device)
    logging.info(f"Test accuracy={acc:.4f} macro_f1={f1:.4f}")
    print(f"accuracy={acc:.4f} macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
