"""
Shape, gradient-flow and cache-equivalence tests for TrafficCLIPFull, in the
same form as A/tests/test_model.py.
"""

import json

import numpy as np
import torch

from A.cache_features_full import build_hypothesis_cache
from A.dataset_full import DYNAMIC_PROMPT_TEMPLATE, build_dynamic_prompts
from A.model_full import TrafficCLIPFull
from transformers import AutoTokenizer

BERT_MODEL_NAME = "google-bert/bert-base-uncased"


def _tiny_batch(n=6, k=4):
    torch.manual_seed(0)
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, k, (n,))
    stats = torch.rand(n, 3) * 10  # (iat, jitter, entropy)-shaped values
    return images, labels, stats


def _tokenize_hypotheses(class_names, stats, max_length=64):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    prompts = build_dynamic_prompts(class_names, stats)
    encoded = tokenizer(
        prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]


def test_forward_shapes():
    k = 4
    class_names = [f"class{i}" for i in range(k)]
    model = TrafficCLIPFull(num_classes=k).eval()
    images, _, stats = _tiny_batch(n=5, k=k)
    input_ids, attention_mask = _tokenize_hypotheses(class_names, stats)

    with torch.no_grad():
        logits_hyp, v_e = model.hypothesis_logits(images, stats, input_ids, attention_mask)

    assert logits_hyp.shape == (5, k)
    assert v_e.shape == (5, 1024)
    assert torch.allclose(v_e.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.isfinite(logits_hyp).all()


def test_gradient_flow_only_through_trainable_modules():
    k = 4
    class_names = [f"class{i}" for i in range(k)]
    model = TrafficCLIPFull(num_classes=k).train()
    images, labels, stats = _tiny_batch(n=6, k=k)
    input_ids, attention_mask = _tokenize_hypotheses(class_names, stats)

    logits_hyp, v_e = model.hypothesis_logits(images, stats, input_ids, attention_mask)
    loss = torch.nn.functional.cross_entropy(logits_hyp, labels)
    loss.backward()

    # logit_scale is inherited from TrafficCLIP but unused by the MLP fusion
    # path, so it is not asserted on here.
    trainable = [
        model.detail_encoder.fc.weight,
        model.semantics_proj.weight,
        model.visual_adapter.down.weight,
        model.text_proj.weight,
        model.stats_proj.net[0].weight,
        model.fusion_head.mlp[0].weight,
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
    model = TrafficCLIPFull(num_classes=4)
    model.train()
    assert model.training is True
    assert model.semantics_encoder.training is False
    assert model.text_encoder.training is False
    assert model.detail_encoder.training is True
    assert model.fusion_head.training is True


def test_hyp_cache_equivalence(tmp_path):
    """
    Correctness gate: if the (N, K, 768) prompt cache does not match a live
    tokenize plus BERT forward for the same (sample, candidate) pairs, every
    number trained against it is wrong.
    """
    n, k = 12, 4
    rng = np.random.RandomState(0)
    x = rng.randn(n, 1, 28, 28).astype(np.float32)
    y = rng.randint(0, k, size=n).astype(np.int64)
    m = rng.rand(n, 3).astype(np.float64) * 10
    labels = np.array([f"class{i}" for i in range(k)], dtype=object)
    npz_path = tmp_path / "tiny.npz"
    np.savez(npz_path, x=x, y=y, m=m, labels=labels)

    cache_dir = tmp_path / "cache_full"
    out_dir = build_hypothesis_cache(npz_path, cache_dir, batch_size=5, seed=62, device=torch.device("cpu"))

    meta = json.loads((out_dir / "meta_hyp.json").read_text())
    assert meta["n"] == n and meta["k"] == k

    hyp = np.load(out_dir / "bert_pooler_hyp.npy")
    assert hyp.shape == (n, k, 768)
    assert np.isfinite(hyp).all()

    model = TrafficCLIPFull(num_classes=k).eval()
    images = torch.from_numpy(x).float()
    stats = torch.from_numpy(m).float()
    idx = torch.arange(n)
    input_ids, attention_mask = _tokenize_hypotheses(list(labels), stats)

    with torch.no_grad():
        logits_live, v_live = model.hypothesis_logits(images, stats, input_ids, attention_mask, idx=None)

    model.load_hyp_text_cache(hyp)
    with torch.no_grad():
        logits_cached, v_cached = model.hypothesis_logits(images, stats, idx=idx)

    assert torch.allclose(v_live, v_cached, atol=1e-5)
    assert torch.allclose(logits_live, logits_cached, atol=1e-4)
