"""
Feature cache consumer.

`semantics_backbone` (ResNet-50) and `text_encoder` (BERT) are frozen and
`no_grad`-wrapped in every model variant, so their outputs are constant per
sample across all epochs, seeds and variants (see `src/cache_features.py`).
This mixin overrides
`get_vision_features` / `get_text_features` to read those outputs from a
cache built by `cache_features.py` when a sample index is supplied, falling
back to the live computation otherwise. Substituting the cache is exact, not
an approximation.

Usage: mix in ahead of the model class so the override wins the MRO.

    class CachedOptimizedTrafficCLIP(CachedFeaturesMixin, OptimizedTrafficCLIP):
        pass

    model = CachedOptimizedTrafficCLIP(num_classes=10, stats_input_dim=3)
    model.load_cache("cache/", "data/processed/ustc_10class.npz")
    v_e = model.get_vision_features(images, idx=batch["idx"])
    t_e = model.get_text_features(idx=batch["idx"], mode="train")
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# get_text_features(mode=...) selects which precomputed tensor to read, and
# therefore which prompt family: --use_stats_prompts and hypothesis testing
# against static-bank matching.
_TEXT_CACHE_MODES = ("train", "hyp", "static")


class CachedFeaturesMixin:
    _cache_f2raw = None
    _cache_bert_train = None
    _cache_bert_hyp = None
    _cache_bert_static = None
    _cache_meta = None

    def load_cache(self, cache_dir, npz_path, mmap=True):
        """Point this model at a cache built by `cache_features.build_cache`."""
        cache_subdir = Path(cache_dir) / Path(npz_path).stem
        meta = json.loads((cache_subdir / "meta.json").read_text())

        mmap_mode = "r" if mmap else None
        self._cache_f2raw = np.load(cache_subdir / "resnet_f2raw.npy", mmap_mode=mmap_mode)
        self._cache_bert_train = np.load(
            cache_subdir / "bert_pooler_train.npy", mmap_mode=mmap_mode
        )
        self._cache_bert_hyp = np.load(
            cache_subdir / "bert_pooler_hyp.npy", mmap_mode=mmap_mode
        )
        self._cache_bert_static = np.load(
            cache_subdir / "bert_pooler_static.npy", mmap_mode=mmap_mode
        )
        self._cache_meta = meta
        logging.info(
            f"Loaded feature cache from {cache_subdir} "
            f"({meta['n']} samples, {meta['k']} classes)"
        )
        return self

    def clear_cache(self):
        self._cache_f2raw = None
        self._cache_bert_train = None
        self._cache_bert_hyp = None
        self._cache_bert_static = None
        self._cache_meta = None

    @property
    def cache_loaded(self):
        return self._cache_f2raw is not None

    def get_vision_features(self, images, idx=None):
        if idx is None or not self.cache_loaded:
            return super().get_vision_features(images)

        # f1 stays live: detail_encoder is trainable, not part of the cache.
        f1 = self.detail_encoder(images)

        f2_raw = self._read_cache(self._cache_f2raw, idx, images.device, images.dtype)
        f2 = self.semantics_proj(f2_raw)
        f2_star = self.alpha * self.visual_adapter(f2) + (1 - self.alpha) * f2

        v_f = torch.cat([f1, f2_star], dim=-1)
        return F.normalize(v_f, p=2, dim=-1)

    def get_text_features(self, input_ids=None, attention_mask=None, idx=None, mode="train"):
        if idx is None or not self.cache_loaded:
            return super().get_text_features(input_ids, attention_mask)
        if mode not in _TEXT_CACHE_MODES:
            raise ValueError(f"mode must be one of {_TEXT_CACHE_MODES}, got {mode!r}")

        cache = {
            "train": self._cache_bert_train,
            "hyp": self._cache_bert_hyp,
            "static": self._cache_bert_static,
        }[mode]

        device = self.text_proj.weight.device
        dtype = self.text_proj.weight.dtype
        pooled = self._read_cache(cache, idx, device, dtype)
        t_f = self.text_proj(pooled)
        return F.normalize(t_f, p=2, dim=-1)

    @staticmethod
    def _read_cache(cache_array, idx, device, dtype):
        idx_np = idx.detach().cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        rows = np.ascontiguousarray(cache_array[idx_np])
        return torch.from_numpy(rows).to(device=device, dtype=dtype)


# Concrete cached model classes used by train_runner.py. Vision is always
# cache-eligible, since idx addresses a sample's ResNet-50 f2_raw regardless
# of model_version or prompt mode. Text is deliberately not threaded through
# forward(), because which cache tensor is correct (train or static) depends
# on args.use_stats_prompts, which forward() does not know. train.py's
# validate() wires the one expensive per-batch text path, hypothesis testing
# over B*K prompts, directly against the "hyp" cache, which is unambiguous:
# it is always keyed by sample idx.
from src.models.opt_traffic_clip import OptimizedTrafficCLIP  # noqa: E402
from src.models.traffic_clip import TrafficCLIP  # noqa: E402


class CachedTrafficCLIP(CachedFeaturesMixin, TrafficCLIP):
    pass


class CachedOptimizedTrafficCLIP(CachedFeaturesMixin, OptimizedTrafficCLIP):
    pass
