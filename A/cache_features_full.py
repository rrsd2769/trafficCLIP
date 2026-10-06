"""
Precomputes the (N, K, 768) per-sample, per-candidate dynamic-prompt BERT
cache used by A/train_full.py.

Running BERT live measured ~7.5s/step at batch 64, K=20, which projects to
roughly 246 hours for 200 epochs x 3 seeds. A sample's stats are fixed for
the whole run, so its K prompt strings are fixed too and only need one pass.
At N=18000, K=20 the cache is ~1.1 GiB of float32, read back mmap'd.

Usage:
    PYTHONPATH=. .venv/bin/python A/cache_features_full.py --npz data/processed/ustc_20class.npz
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from A.cache_features import get_device, set_seed
from A.dataset_full import DYNAMIC_PROMPT_TEMPLATE, TrafficStatsDataset
from A.model import BERT_MODEL_NAME
from A.train_full import MAX_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def build_hypothesis_cache(npz_path, cache_dir, batch_size=32, seed=62, device=None):
    """
    For every real sample i and every candidate class name j, embeds
    "...class {j} with {sample i's own real stats}." and stores the pooled
    BERT output at hyp_cache[i, j]. `batch_size` here is samples per chunk
    (each chunk runs batch_size * K sequences through BERT in one call).
    """
    npz_path = Path(npz_path)
    device = device or get_device()
    set_seed(seed)

    ds = TrafficStatsDataset(npz_path)
    n, k = len(ds), len(ds.class_names)
    logging.info(f"{npz_path.name}: {n} samples x {k} classes = {n * k} prompts -> device={device}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    text_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device).eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    out_dir = Path(cache_dir) / npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp = np.lib.format.open_memmap(
        out_dir / "bert_pooler_hyp.npy", mode="w+", dtype=np.float32, shape=(n, k, 768)
    )

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            stats_chunk = ds.stats[start:end].numpy()  # (b, 3)
            b = end - start
            prompts = [
                DYNAMIC_PROMPT_TEMPLATE.format(
                    name=name.replace("_", " "), iat=stats_chunk[i, 0], jitter=stats_chunk[i, 1], entropy=stats_chunk[i, 2]
                )
                for i in range(b)
                for name in ds.class_names
            ]
            encoded = tokenizer(
                prompts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(device)
            pooled = text_encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).pooler_output
            hyp[start:end] = pooled.float().cpu().numpy().reshape(b, k, 768)

            if (start // batch_size) % 20 == 0:
                elapsed = time.time() - t0
                done = end
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                logging.info(f"{done}/{n} samples ({100*done/n:.1f}%), {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    hyp.flush()
    meta = {
        "npz_path": str(npz_path),
        "n": n,
        "k": k,
        "labels": ds.class_names,
        "tokenizer": BERT_MODEL_NAME,
        "max_length": MAX_LENGTH,
        "dynamic_template": DYNAMIC_PROMPT_TEMPLATE,
        "shape": [n, k, 768],
    }
    (out_dir / "meta_hyp.json").write_text(json.dumps(meta, indent=2))
    logging.info(f"bert_pooler_hyp: {hyp.shape} in {time.time() - t0:.1f}s -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build A-Full's per-sample dynamic-prompt BERT cache.")
    parser.add_argument("--npz", required=True, help="Path to the .npz archive to cache.")
    parser.add_argument("--cache-dir", default="A/cache_full")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per chunk (x K prompts each).")
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()

    build_hypothesis_cache(npz_path=args.npz, cache_dir=args.cache_dir, batch_size=args.batch_size, seed=args.seed)
