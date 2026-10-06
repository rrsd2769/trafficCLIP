"""
Precompute the frozen-backbone outputs once per .npz archive.

`semantics_backbone` (ResNet-50, keeping its randomly initialised and frozen
`conv1`) and `text_encoder` (BERT) never receive a gradient in any model
variant, so their outputs are constant per sample across every epoch, seed and
variant, and precomputing them is exact rather than approximate.
`src/models/cached_mixin.py` consumes what this script writes.

`set_seed(seed)` runs before the model is built so the frozen `conv1` draw
here matches the draw a training run makes with the same seed, which
`tests/test_cache_equivalence.py` depends on.

Four tensors are written per archive, under `cache/<archive-stem>/`:

    resnet_f2raw.npy        (N, 2048)     ResNet-50 pool output, one per sample
    bert_pooler_static.npy  (K, 768)      class-name-only prompt, one per class
    bert_pooler_train.npy   (N, 768)      dynamic prompt using the sample's own
                                           label and stats, read by training
    bert_pooler_hyp.npy     (N, K, 768)   dynamic prompt using the sample's own
                                           stats against every candidate class
                                           name, read by hypothesis-testing eval

Usage:

    PYTHONPATH=. .venv/bin/python src/cache_features.py --npz data/processed/ustc_20class.npz
    PYTHONPATH=. .venv/bin/python src/cache_features.py --npz data/processed/ustc_10class.npz
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models.traffic_clip import TrafficCLIP
from src.utils.utils import get_device, load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

STATIC_TEMPLATE = "A network traffic gray photo of class {name}."
DYNAMIC_TEMPLATE = (
    "A network traffic gray photo of class {name} with "
    "{iat:.2f}ms mean IAT, {jitter:.2f}ms jitter, and {entropy:.2f} byte entropy."
)


def _bert_pooler_batched(text_encoder, tokenizer, prompts, max_length, device, batch_size):
    """Runs frozen BERT over `prompts` in batches, returns (len(prompts), 768) float32."""
    out = np.empty((len(prompts), 768), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            pooled = text_encoder(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            ).pooler_output
            out[start : start + len(chunk)] = pooled.float().cpu().numpy()
    return out


def build_cache(
    npz_path,
    cache_dir,
    tokenizer_name="google-bert/bert-base-uncased",
    max_length=64,
    batch_size=256,
    seed=62,
    device=None,
):
    npz_path = Path(npz_path)
    device = device or get_device()

    with np.load(npz_path, allow_pickle=True) as d:
        x = d["x"]
        y = d["y"]
        m = d["m"]
        labels = list(d["labels"])

    n, k = len(x), len(labels)
    logging.info(f"{npz_path.name}: {n} samples, {k} classes -> device={device}")

    set_seed(seed)
    model = TrafficCLIP().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    out_dir = Path(cache_dir) / npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- resnet_f2raw: (N, 2048) ---
    t0 = time.time()
    f2raw = np.empty((n, 2048), dtype=np.float32)
    images = torch.from_numpy(x).float()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = images[start : start + batch_size].to(device)
            feat = model.semantics_backbone(batch).flatten(1)
            f2raw[start : start + len(batch)] = feat.float().cpu().numpy()
    np.save(out_dir / "resnet_f2raw.npy", f2raw)
    logging.info(f"resnet_f2raw: {f2raw.shape} in {time.time() - t0:.1f}s")

    # --- bert_pooler_static: (K, 768) ---
    t0 = time.time()
    static_prompts = [STATIC_TEMPLATE.format(name=name) for name in labels]
    static_pooler = _bert_pooler_batched(
        model.text_encoder, tokenizer, static_prompts, max_length, device, batch_size
    )
    np.save(out_dir / "bert_pooler_static.npy", static_pooler)
    logging.info(f"bert_pooler_static: {static_pooler.shape} in {time.time() - t0:.1f}s")

    # --- bert_pooler_train: (N, 768) ---
    t0 = time.time()
    train_prompts = [
        DYNAMIC_TEMPLATE.format(name=labels[y[i]], iat=m[i, 0], jitter=m[i, 1], entropy=m[i, 2])
        for i in range(n)
    ]
    train_pooler = _bert_pooler_batched(
        model.text_encoder, tokenizer, train_prompts, max_length, device, batch_size
    )
    np.save(out_dir / "bert_pooler_train.npy", train_pooler)
    logging.info(f"bert_pooler_train: {train_pooler.shape} in {time.time() - t0:.1f}s")

    # --- bert_pooler_hyp: (N, K, 768) ---
    t0 = time.time()
    hyp_path = out_dir / "bert_pooler_hyp.npy"
    hyp_mm = np.lib.format.open_memmap(hyp_path, mode="w+", dtype=np.float32, shape=(n, k, 768))
    flat = hyp_mm.reshape(n * k, 768)
    hyp_prompts = [
        DYNAMIC_TEMPLATE.format(name=name, iat=m[i, 0], jitter=m[i, 1], entropy=m[i, 2])
        for i in range(n)
        for name in labels
    ]
    with torch.no_grad():
        for start in range(0, len(hyp_prompts), batch_size):
            chunk = hyp_prompts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            pooled = model.text_encoder(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            ).pooler_output
            flat[start : start + len(chunk)] = pooled.float().cpu().numpy()
    hyp_mm.flush()
    del hyp_mm, flat
    logging.info(f"bert_pooler_hyp: ({n}, {k}, 768) in {time.time() - t0:.1f}s")

    meta = {
        "npz_path": str(npz_path),
        "n": n,
        "k": k,
        "labels": labels,
        "seed": seed,
        "tokenizer": tokenizer_name,
        "max_length": max_length,
        "static_template": STATIC_TEMPLATE,
        "dynamic_template": DYNAMIC_TEMPLATE,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.info(f"Cache written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the frozen-backbone feature cache.")
    parser.add_argument("--npz", required=True, help="Path to the .npz archive to cache.")
    parser.add_argument("--cache-dir", default=None, help="Defaults to config paths.cache_dir.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=62, help="Must match training's set_seed().")
    args = parser.parse_args()

    config = load_config()
    cache_dir = args.cache_dir or config["paths"]["cache_dir"]
    build_cache(
        npz_path=args.npz,
        cache_dir=cache_dir,
        tokenizer_name=config["preprocess"]["tokenizer"],
        max_length=config["preprocess"]["max_length"],
        batch_size=args.batch_size,
        seed=args.seed,
    )
