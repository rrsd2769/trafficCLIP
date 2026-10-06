# Feature cache

`semantics_backbone` (ResNet-50, `src/models/traffic_clip.py`) and
`text_encoder` (BERT) are frozen and `no_grad`-wrapped in every model variant,
`TrafficCLIP` and `OptimizedTrafficCLIP` alike. `TrafficDataset` returns images
with no augmentation, so their outputs are **constant per sample** across every
epoch, seed and variant for as long as the backbones stay frozen. Precomputing
them once is exact, not an approximation. Only `semantics_proj`,
`visual_adapter`, `detail_encoder`, `text_proj` and whatever fusion head sits on
top actually train.

## Building a cache

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. .venv/bin/python src/cache_features.py --npz data/processed/ustc_10class.npz
HF_HUB_DISABLE_XET=1 PYTHONPATH=. .venv/bin/python src/cache_features.py --npz data/processed/ustc_20class.npz
```

See `docs/ENVIRONMENT.md` for why `HF_HUB_DISABLE_XET=1` is needed against a
cold Hugging Face cache. `--seed` defaults to **62**, matching the seed every
training and eval run in this project uses. It has to match: B's frozen `conv1`
is drawn randomly at model construction, so the cache is only valid for a model
built from the same RNG state. `tests/test_cache_equivalence.py` catches a
mismatch.

## Layout

One directory per archive, under `cache/<archive-stem>/`:

| Tensor | Shape (10-class / 20-class) | Built from | Read by |
|---|---|---|---|
| `resnet_f2raw.npy` | `(9000, 2048)` / `(18000, 2048)` | `semantics_backbone(images)` | `get_vision_features(images, idx=...)` |
| `bert_pooler_static.npy` | `(10, 768)` / `(20, 768)` | class-name-only prompt, one per class | `get_text_features(idx=class_idx, mode="static")`, the static descriptor bank path |
| `bert_pooler_train.npy` | `(9000, 768)` / `(18000, 768)` | sample's own label and own stats | `get_text_features(idx=sample_idx, mode="train")`, training's per-sample prompt |
| `bert_pooler_hyp.npy` | `(9000, 10, 768)` / `(18000, 20, 768)` | sample's own stats, every candidate class name | `get_text_features(idx=sample_idx, mode="hyp")`, hypothesis-testing eval (`validate()`'s per-sample K-way loop) |

Plus `meta.json`: source npz path, `n`, `k`, `labels`, and the seed and prompt
templates used, for checking a loaded cache against the model it is paired with.

`bert_pooler_hyp` is the expensive one, at `N * K` BERT forward passes, one per
`(sample, candidate class)` pair, because the hypothesis-testing prediction rule
keeps the sample's stats fixed and varies only the class name (see `validate()`
in `src/train.py`). Measured build times on the M4 (MPS, batch 256):

| Archive | `resnet_f2raw` | `bert_pooler_train` | `bert_pooler_hyp` | Total | On disk |
|---|---|---|---|---|---|
| `ustc_10class` | 5.5s | 43.3s | 497.5s (~8.3 min) | ~9.5 min | 360 MB |
| `ustc_20class` | 9.7s | 100.4s | 2156.2s (~36 min) | ~36.5 min | 1.2 GB |

## Consuming a cache

`src/models/cached_mixin.py` provides `CachedFeaturesMixin`, mixed in ahead of
`TrafficCLIP` or `OptimizedTrafficCLIP`:

```python
class CachedOptimizedTrafficCLIP(CachedFeaturesMixin, OptimizedTrafficCLIP):
    pass

model = CachedOptimizedTrafficCLIP(num_classes=10, stats_input_dim=3)
model.load_cache("cache/", "data/processed/ustc_10class.npz")

v_e = model.get_vision_features(images, idx=batch["idx"])
t_e = model.get_text_features(idx=batch["idx"], mode="train")       # training
t_e = model.get_text_features(idx=batch["idx"], mode="hyp")         # eval, per-sample hypothesis bank -> (B, K, 1024) once reshaped
t_e = model.get_text_features(idx=class_idx, mode="static")         # static descriptor bank
```

`idx=None`, the default, falls through to the live computation unchanged, so
call sites that do not pass `idx` are unaffected. `TrafficDataset.__getitem__`
(`src/dataset.py`) returns `"idx"` alongside the existing keys so a batch can
address the cache.

`scripts/benchmark_cache_epoch.py` builds a minimal training loop directly on
the mixin, and is where the epoch timings below come from.

## Checks

- `tests/test_cache_equivalence.py`: cached and live agree to `atol=1e-5` for
  vision features and all three text-cache modes, using real ResNet-50 and BERT
  weights over tiny synthetic data. If this fails the cache is invalid and every
  downstream number is wrong.
- `scripts/benchmark_cache_epoch.py --npz data/processed/ustc_10class.npz`: one
  training epoch, 6,300 samples at batch 64, over the trainable submodules only.
  Measured on the M4 at **2.5s cached against 38.2s live**, roughly 15x at this
  batch and model scale. The gap widens at full scale (100 epochs, 5 seeds, the
  20-class matrix), where a live epoch runs 2 to 5 minutes.

## A live epoch needs an explicit `eval()` on the frozen submodules

`src/train.py`'s `train()` calls `model.train()` on the whole model, and nothing
separately pins `semantics_backbone` or `text_encoder` to eval mode. Only their
*parameters* are frozen, via `requires_grad=False`, which is orthogonal to
`.training`. Two consequences:

- **BERT's dropout stays active.** `get_text_features` does not wrap the
  `text_encoder` call in `no_grad` either, so a live forward pass in training
  mode is stochastic per call and the same sample's pooler output differs from
  epoch to epoch. On MPS it does not merely drift, it crashes:
  `torch.nn.functional.scaled_dot_product_attention for MPS does not support
  dropout` (`NotImplementedError`), reproduced directly by
  `scripts/benchmark_cache_epoch.py`. CPU and CUDA see the drift without the
  crash.
- **ResNet-50's BatchNorm running stats keep updating.** The `no_grad` around
  `semantics_backbone(images)` (`traffic_clip.py:134`) blocks gradients, not
  buffer updates: `running_mean` and `running_var` are plain tensor ops that
  still execute in train mode.

Both undercut the premise that the frozen backbones are constant, and the second
is already true of the original training loop, silently.
`scripts/benchmark_cache_epoch.py` works around it by calling
`model.semantics_backbone.eval()` and `model.text_encoder.eval()` right after
`model.train()`, and `src/train.py`'s `train()` now does the same. On this
hardware that is not an optimization, it is the difference between training at
all and crashing on the first live batch. The cache sidesteps the problem
entirely, because once `idx` is supplied the cached paths never call
`text_encoder` or `semantics_backbone`.

## `_t60` variants

The `--flow-timeout 60` archives (`ustc_{10,20}class_t60.npz`) need their own
cache if an experiment trains on them. Both `resnet_f2raw` (the images differ,
since flow segmentation differs even though masking does not) and
`bert_pooler_*` (the dynamic-prompt stats differ) are archive-specific.
`cache_features.py` keys off the npz stem, so this falls out for free:

```bash
PYTHONPATH=. .venv/bin/python src/cache_features.py --npz data/processed/ustc_10class_t60.npz
```
