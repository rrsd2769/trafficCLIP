"""
Times one training epoch with the feature cache enabled against the live
backbones. Trains only the trainable submodules (detail_encoder,
semantics_proj, visual_adapter, text_proj, fusion_head) over the 10-class
train split, using CachedFeaturesMixin rather than running ResNet-50 and
BERT per batch.

Pass --skip-live to time only the cached path:

    PYTHONPATH=. .venv/bin/python scripts/benchmark_cache_epoch.py --npz data/processed/ustc_10class.npz
"""

import argparse
import logging
import time

import torch
import torch.nn as nn

from src.dataset import get_dataloader
from src.models.cached_mixin import CachedFeaturesMixin
from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.utils.utils import get_device, load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class CachedOptimizedTrafficCLIP(CachedFeaturesMixin, OptimizedTrafficCLIP):
    pass


def _run_epoch(model, train_loader, device, use_cache):
    model.train()
    # semantics_backbone and text_encoder are frozen and must stay in eval
    # mode: nothing else in this codebase pins them there, so a bare
    # model.train() leaves BERT's dropout active. On MPS that combination
    # (dropout + the fused SDPA attention kernel) is unimplemented and
    # crashes outright; on any backend it also means the "frozen" pooler
    # output is stochastic per call and BatchNorm running stats keep
    # drifting, both of which contradict the constant-per-sample premise
    # the cache relies on. See docs/feature_cache.md.
    model.semantics_backbone.eval()
    model.text_encoder.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    t0 = time.time()
    for batch in train_loader:
        images = batch["image"].to(device)
        stats = batch["stats"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        if use_cache:
            idx = batch["idx"].to(device)
            v_e = model.get_vision_features(images, idx=idx)
            t_e = model.get_text_features(idx=idx, mode="train")
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            v_e = model.get_vision_features(images)
            t_e = model.get_text_features(input_ids, attention_mask)

        s_e = model.stats_proj(stats)
        s_e = torch.nn.functional.normalize(s_e, p=2, dim=-1)
        combined = torch.cat((v_e, t_e, s_e), dim=1)
        logits = model.fusion_head(combined)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    return time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Time a cached training epoch against the live backbones.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--skip-live", action="store_true", help="Only time the cached epoch.")
    args = parser.parse_args()

    config = load_config()
    cache_dir = args.cache_dir or config["paths"]["cache_dir"]
    device = get_device()

    set_seed(args.seed)
    train_loader, _, _ = get_dataloader(
        npz_path=args.npz,
        tokenizer=config["preprocess"]["tokenizer"],
        batch_size=args.batch_size,
        max_length=config["preprocess"]["max_length"],
        seed=args.seed,
        use_dynamic_prompts=True,
    )
    num_classes = len(train_loader.dataset.dataset.class_names)

    if not args.skip_live:
        set_seed(args.seed)
        live_model = OptimizedTrafficCLIP(
            num_classes=num_classes, stats_input_dim=3, use_stats=True
        ).to(device)
        live_time = _run_epoch(live_model, train_loader, device, use_cache=False)
        logging.info(f"LIVE   epoch: {live_time:.1f}s")
        del live_model

    set_seed(args.seed)
    cached_model = CachedOptimizedTrafficCLIP(
        num_classes=num_classes, stats_input_dim=3, use_stats=True
    ).to(device)
    cached_model.load_cache(cache_dir, args.npz)
    cached_time = _run_epoch(cached_model, train_loader, device, use_cache=True)
    logging.info(f"CACHED epoch: {cached_time:.1f}s")

    if cached_time >= 60:
        logging.error("FAILED: cached epoch did not complete under 60s")
    else:
        logging.info("PASSED: cached epoch under 60s")


if __name__ == "__main__":
    main()
