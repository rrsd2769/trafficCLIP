"""
Precomputes the frozen semantics-encoder outputs and the K static per-class
BERT prompt embeddings.

The text side is one static prompt per class, so the whole text cache is a
single (K, 768) tensor rather than one embedding per training sample.

resnet_f2raw is not interchangeable with the cache under cache/: the conv1
kernels differ, so every value of the (N, 2048) tensor differs.
--broken-conv1 writes the ablation variant to its own subdirectory.

Usage:
    PYTHONPATH=. .venv/bin/python A/cache_features.py --npz data/processed/ustc_20class.npz
    PYTHONPATH=. .venv/bin/python A/cache_features.py --npz data/processed/ustc_20class.npz --broken-conv1
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from A.model import BERT_MODEL_NAME, STATIC_PROMPT_TEMPLATE, TrafficCLIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_cache(
    npz_path,
    cache_dir,
    broken_conv1=False,
    max_length=32,
    batch_size=256,
    seed=62,
    device=None,
):
    npz_path = Path(npz_path)
    device = device or get_device()

    with np.load(npz_path, allow_pickle=True) as d:
        x = d["x"]
        labels = list(d["labels"])

    n, k = len(x), len(labels)
    logging.info(
        f"{npz_path.name}: {n} samples, {k} classes -> device={device}, "
        f"broken_conv1={broken_conv1}"
    )

    # Only matters under --broken-conv1: the random conv1 draw has to be
    # reproducible so the cache matches a live model built with the same seed.
    set_seed(seed)
    model = TrafficCLIP(broken_conv1=broken_conv1).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    variant = "broken_conv1" if broken_conv1 else "correct_conv1"
    out_dir = Path(cache_dir) / npz_path.stem / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- resnet_f2raw: (N, 2048) ---
    t0 = time.time()
    f2raw = np.empty((n, 2048), dtype=np.float32)
    images = torch.from_numpy(x).float()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = images[start : start + batch_size].to(device)
            feat = model.semantics_encoder(batch)
            f2raw[start : start + len(batch)] = feat.float().cpu().numpy()
    np.save(out_dir / "resnet_f2raw.npy", f2raw)
    logging.info(f"resnet_f2raw: {f2raw.shape} in {time.time() - t0:.1f}s")

    # --- bert_pooler_static: (K, 768) ---
    t0 = time.time()
    prompts = [STATIC_PROMPT_TEMPLATE.format(name=name.replace("_", " ")) for name in labels]
    enc = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        pooled = model.text_encoder(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).pooler_output
    bert_static = pooled.float().cpu().numpy()
    np.save(out_dir / "bert_pooler_static.npy", bert_static)
    logging.info(f"bert_pooler_static: {bert_static.shape} in {time.time() - t0:.1f}s")

    meta = {
        "npz_path": str(npz_path),
        "n": n,
        "k": k,
        "labels": labels,
        "seed": seed,
        "broken_conv1": broken_conv1,
        "tokenizer": BERT_MODEL_NAME,
        "max_length": max_length,
        "static_template": STATIC_PROMPT_TEMPLATE,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.info(f"Cache written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build A's frozen-encoder feature cache.")
    parser.add_argument("--npz", required=True, help="Path to the .npz archive to cache.")
    parser.add_argument("--cache-dir", default="A/cache")
    parser.add_argument(
        "--broken-conv1",
        action="store_true",
        help="Ablation: cache the random-frozen-conv1 variant instead.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()

    build_cache(
        npz_path=args.npz,
        cache_dir=args.cache_dir,
        broken_conv1=args.broken_conv1,
        batch_size=args.batch_size,
        seed=args.seed,
    )
