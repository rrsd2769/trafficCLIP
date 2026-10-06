"""
The cached and live feature paths must agree to atol=1e-5.

`semantics_backbone` and `text_encoder` are frozen and `no_grad`-wrapped in
every model variant, so substituting precomputed values for their outputs is
exact. This builds a small real cache with `cache_features.build_cache`, using
real ResNet-50 and BERT weights over tiny synthetic data, and checks that
`CachedFeaturesMixin` reads back what the live model computes, for vision
features and for all three text-cache modes (train, hyp, static).

If this fails, the cache is invalid and every downstream number is wrong.
"""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from src.cache_features import DYNAMIC_TEMPLATE, STATIC_TEMPLATE, build_cache
from src.models.cached_mixin import CachedFeaturesMixin
from src.models.traffic_clip import TrafficCLIP
from src.utils.utils import set_seed

SEED = 62
N_SAMPLES = 48  # keeps bert_pooler_hyp small (N * K forward passes)
N_CLASSES = 4


class CachedTrafficCLIP(CachedFeaturesMixin, TrafficCLIP):
    pass


@pytest.fixture(scope="module")
def synthetic_npz(tmp_path_factory):
    rng = np.random.default_rng(0)
    labels = ["Alpha", "Bravo", "Charlie", "Delta"][:N_CLASSES]

    x = rng.random((N_SAMPLES, 1, 28, 28), dtype=np.float32)
    y = np.array([i % N_CLASSES for i in range(N_SAMPLES)], dtype=np.int64)
    m = rng.uniform(0, 500, size=(N_SAMPLES, 3)).astype(np.float32)

    path = tmp_path_factory.mktemp("data") / "synthetic.npz"
    np.savez_compressed(path, x=x, y=y, m=m, labels=labels)
    return path, labels, x, y, m


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory, synthetic_npz):
    npz_path, *_ = synthetic_npz
    out = tmp_path_factory.mktemp("cache")
    build_cache(
        npz_path=npz_path,
        cache_dir=out,
        max_length=32,
        batch_size=16,
        seed=SEED,
        device=torch.device("cpu"),
    )
    return out


@pytest.fixture(scope="module")
def model():
    set_seed(SEED)
    return CachedTrafficCLIP().to("cpu").eval()


def test_vision_features_match(model, cache_dir, synthetic_npz):
    npz_path, *_ = synthetic_npz
    images = torch.from_numpy(np.load(npz_path)["x"]).float()

    with torch.no_grad():
        live = model.get_vision_features(images)

    model.load_cache(cache_dir, npz_path, mmap=False)
    idx = torch.arange(len(images))
    with torch.no_grad():
        cached = model.get_vision_features(images, idx=idx)
    model.clear_cache()

    torch.testing.assert_close(live, cached, atol=1e-5, rtol=0)


def test_text_features_train_mode_match(model, cache_dir, synthetic_npz):
    npz_path, labels, x, y, m = synthetic_npz
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    prompts = [
        DYNAMIC_TEMPLATE.format(name=labels[y[i]], iat=m[i, 0], jitter=m[i, 1], entropy=m[i, 2])
        for i in range(N_SAMPLES)
    ]
    enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")

    with torch.no_grad():
        live = model.get_text_features(enc["input_ids"], enc["attention_mask"])

    model.load_cache(cache_dir, npz_path, mmap=False)
    idx = torch.arange(N_SAMPLES)
    with torch.no_grad():
        cached = model.get_text_features(idx=idx, mode="train")
    model.clear_cache()

    torch.testing.assert_close(live, cached, atol=1e-5, rtol=0)


def test_text_features_static_mode_match(model, cache_dir, synthetic_npz):
    npz_path, labels, *_ = synthetic_npz
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    prompts = [STATIC_TEMPLATE.format(name=name) for name in labels]
    enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")

    with torch.no_grad():
        live = model.get_text_features(enc["input_ids"], enc["attention_mask"])

    model.load_cache(cache_dir, npz_path, mmap=False)
    idx = torch.arange(len(labels))
    with torch.no_grad():
        cached = model.get_text_features(idx=idx, mode="static")
    model.clear_cache()

    torch.testing.assert_close(live, cached, atol=1e-5, rtol=0)


def test_text_features_hyp_mode_matches_per_sample_hypothesis_bank(model, cache_dir, synthetic_npz):
    """
    validate()'s hypothesis-testing path builds K prompts per sample -- the
    sample's own stats, every candidate class name -- and runs BERT once per
    sample. bert_pooler_hyp precomputes exactly that.
    """
    npz_path, labels, x, y, m = synthetic_npz
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    model.load_cache(cache_dir, npz_path, mmap=False)

    # A handful of samples rather than all 48: this is O(N*K) live BERT calls.
    for i in (0, 1, N_SAMPLES // 2, N_SAMPLES - 1):
        sample_prompts = [
            DYNAMIC_TEMPLATE.format(name=name, iat=m[i, 0], jitter=m[i, 1], entropy=m[i, 2])
            for name in labels
        ]
        enc = tokenizer(
            sample_prompts, padding="max_length", truncation=True, max_length=32, return_tensors="pt"
        )
        with torch.no_grad():
            live = model.get_text_features(enc["input_ids"], enc["attention_mask"])
            cached = model.get_text_features(idx=torch.tensor([i]), mode="hyp")

        torch.testing.assert_close(live, cached.squeeze(0), atol=1e-5, rtol=0)

    model.clear_cache()


def test_fallback_to_live_when_no_idx(model):
    """idx=None must behave exactly like the un-mixed-in base class (no cache required)."""
    images = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        out = model.get_vision_features(images)
    assert out.shape == (4, 1024)
    assert not model.cache_loaded
