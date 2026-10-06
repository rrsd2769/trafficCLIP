"""
Shape, gradient-flow, conv1-adaptation and cache-equivalence tests for
A/model.py. Runs real pretrained ResNet-50/BERT weights against tiny
synthetic batches, so it does not need the USTC archives.
"""

import json

import numpy as np
import torch

from A.cache_features import build_cache
from A.model import TrafficCLIP, adapt_conv1_to_grayscale
from torchvision.models import ResNet50_Weights, resnet50


def _tiny_batch(n=6, k=4):
    torch.manual_seed(0)
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, k, (n,))
    return images, labels


def test_forward_shapes():
    torch.manual_seed(0)
    model = TrafficCLIP().eval()
    images, _ = _tiny_batch(n=5, k=4)
    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)

    logits, v_e, t_e, scale = model(images, input_ids=input_ids, attention_mask=attention_mask)

    assert logits.shape == (5, 4)
    assert v_e.shape == (5, 1024)
    assert t_e.shape == (4, 1024)
    assert scale.dim() == 0
    # L2-normalized rows
    assert torch.allclose(v_e.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(t_e.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_gradient_flow_only_through_trainable_modules():
    model = TrafficCLIP().train()
    images, labels = _tiny_batch(n=6, k=4)
    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)

    logits, v_e, _, _ = model(images, input_ids=input_ids, attention_mask=attention_mask)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    trainable = [
        model.detail_encoder.fc.weight,
        model.semantics_proj.weight,
        model.visual_adapter.down.weight,
        model.text_proj.weight,
        model.logit_scale,
    ]
    for p in trainable:
        assert p.grad is not None and torch.isfinite(p.grad).all()

    for p in model.semantics_encoder.backbone.parameters():
        assert p.requires_grad is False
        assert p.grad is None
    for p in model.text_encoder.parameters():
        assert p.requires_grad is False
        assert p.grad is None


def test_train_mode_pins_frozen_submodules_to_eval():
    model = TrafficCLIP()
    model.train()
    assert model.training is True
    assert model.semantics_encoder.training is False
    assert model.text_encoder.training is False
    assert model.detail_encoder.training is True


def test_conv1_adaptation_matches_replicated_grayscale_forward():
    """
    Summing the pretrained RGB conv1 kernel across channels must give the
    same output as feeding a 3-channel replica of the same grayscale image
    through the original conv1.
    """
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    original_conv1 = resnet.conv1
    adapted_conv1 = adapt_conv1_to_grayscale(original_conv1)

    torch.manual_seed(0)
    gray = torch.randn(2, 1, 64, 64)
    replicated = gray.repeat(1, 3, 1, 1)

    with torch.no_grad():
        out_adapted = adapted_conv1(gray)
        out_replicated = original_conv1(replicated)

    assert torch.allclose(out_adapted, out_replicated, atol=1e-4)


def test_broken_conv1_ablation_flag_actually_changes_the_backbone():
    torch.manual_seed(0)
    correct = TrafficCLIP(broken_conv1=False)
    torch.manual_seed(0)
    broken = TrafficCLIP(broken_conv1=True)

    correct_w = correct.semantics_encoder.backbone[0].weight
    broken_w = broken.semantics_encoder.backbone[0].weight
    assert correct_w.shape == broken_w.shape
    assert not torch.allclose(correct_w, broken_w)


def test_cache_equivalence(tmp_path):
    """
    Correctness gate: if the cached vision/text features do not match a live
    forward pass, every number trained against the cache is wrong.
    """
    n, k = 48, 4
    rng = np.random.RandomState(0)
    x = rng.randn(n, 1, 28, 28).astype(np.float32)
    y = rng.randint(0, k, size=n).astype(np.int64)
    labels = np.array([f"class{i}" for i in range(k)], dtype=object)
    npz_path = tmp_path / "tiny.npz"
    np.savez(npz_path, x=x, y=y, labels=labels)

    cache_dir = tmp_path / "cache"
    out_dir = build_cache(npz_path, cache_dir, broken_conv1=False, seed=62, device=torch.device("cpu"))

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n"] == n and meta["k"] == k

    f2raw = np.load(out_dir / "resnet_f2raw.npy")
    bert_static = np.load(out_dir / "bert_pooler_static.npy")
    assert f2raw.shape == (n, 2048)
    assert bert_static.shape == (k, 768)
    assert np.isfinite(f2raw).all()
    assert np.isfinite(bert_static).all()

    import random as _random

    _random.seed(62)
    np.random.seed(62)
    torch.manual_seed(62)
    model = TrafficCLIP(broken_conv1=False).eval()

    images = torch.from_numpy(x).float()
    idx = torch.arange(n)
    with torch.no_grad():
        v_live = model.get_vision_features(images, idx=None)
        v_cached = model.get_vision_features(images, idx=idx)
    model.load_vision_cache(f2raw)
    with torch.no_grad():
        v_from_cache = model.get_vision_features(images, idx=idx)

    assert torch.allclose(v_live, v_cached, atol=1e-5)
    assert torch.allclose(v_live, v_from_cache, atol=1e-5)

    model.load_text_cache(bert_static)
    with torch.no_grad():
        t_from_cache = model.get_text_features()
    assert t_from_cache.shape == (k, 1024)
    assert torch.allclose(t_from_cache.norm(dim=-1), torch.ones(k), atol=1e-5)
