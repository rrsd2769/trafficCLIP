"""
Independent implementation of Chen et al., "TrafficCLIP: A lightweight
cross-modal framework for network traffic classification", Computer Networks
272 (2025) 111662, written from Algorithm 1 and Figs. 2-4 of that paper. It
shares no code with src/models/.

Three points where this differs from src/models/traffic_clip.py:

  - conv1 is adapted from the pretrained RGB kernel by summing across the
    input-channel dimension rather than replaced by a fresh random Conv2d,
    so the pretrained ResNet-50 sees real input through its first layer.
  - Text is the paper's static per-class hard prompt only ("A network
    traffic gray photo of {class}"). The per-sample prompt keyed off
    IAT/jitter/entropy is not in the base paper and is not implemented here.
  - train() pins the frozen submodules to eval mode itself rather than
    leaving it to each call site. Otherwise BERT's dropout stays active
    during the forward pass (a hard crash on MPS, where
    scaled_dot_product_attention has no dropout kernel) and ResNet's
    BatchNorm running stats keep drifting, since requires_grad=False freezes
    parameters and not .training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel

BERT_MODEL_NAME = "google-bert/bert-base-uncased"

# Sec. 4.2's hard prompt template. The paper spells it "gray" in the body
# text and "grey" in Fig. 1c; "gray" also matches src/dataset.py:61, so the
# only wording difference between the two models is the added dynamic clause.
STATIC_PROMPT_TEMPLATE = "A network traffic gray photo of {name}."


class ResidualBlock(nn.Module):
    """1x1 (reduce) -> 3x3 (spatial) -> 1x1 (restore) residual block, Fig. 3."""

    def __init__(self, channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class DetailAwareEncoder(nn.Module):
    """Fig. 3: initial 3x3 conv -> residual blocks -> global pool -> FC -> f1 (512-d)."""

    def __init__(self, num_blocks=2, channels=64, mid_channels=16, out_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, mid_channels) for _ in range(num_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def adapt_conv1_to_grayscale(conv1: nn.Conv2d) -> nn.Conv2d:
    """
    Sum the pretrained RGB conv1 kernel across the input-channel dimension to
    get a single-channel kernel that takes grayscale input directly.

    Identical to replicating a grayscale value G across R/G/B and convolving
    with the original 3-channel kernel, since sum_c(W[o,c] * G) ==
    (sum_c W[o,c]) * G, but without tripling the forward cost.
    """
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        new_conv1.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))
        if conv1.bias is not None:
            new_conv1.bias.copy_(conv1.bias)
    return new_conv1


class SemanticsAwareEncoder(nn.Module):
    """Pretrained ResNet-50, adapted for 1-channel input, entirely frozen."""

    def __init__(self, broken_conv1=False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        if broken_conv1:
            # Ablation only: a fresh, randomly initialised, permanently
            # frozen conv1, to measure what that initialisation costs.
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            resnet.conv1 = adapt_conv1_to_grayscale(resnet.conv1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x).flatten(1)


class TrafficAdapter(nn.Module):
    """Eq. 3, Fig. 4: down-projection -> ReLU -> up-projection."""

    def __init__(self, embed_dim=512, bottleneck_dim=256):
        super().__init__()
        self.down = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, embed_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, f2):
        return self.up(self.relu(self.down(f2)))


class TrafficCLIP(nn.Module):
    """TrafficCLIP. See the module docstring for the implementation choices."""

    def __init__(self, alpha=0.9, broken_conv1=False, num_detail_blocks=2):
        super().__init__()
        self.alpha = alpha

        self.detail_encoder = DetailAwareEncoder(num_blocks=num_detail_blocks)

        self.semantics_encoder = SemanticsAwareEncoder(broken_conv1=broken_conv1)
        self.semantics_proj = nn.Linear(2048, 512)
        self.visual_adapter = TrafficAdapter(embed_dim=512)

        self.text_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(768, 1024)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        # Populated by load_vision_cache / load_text_cache. Both None means
        # every forward pass computes live.
        self._f2raw_cache = None
        self._bert_static_cache = None

    def train(self, mode=True):
        super().train(mode)
        # Frozen submodules must never leave eval mode: requires_grad=False
        # blocks gradients, not BERT's dropout or ResNet's BatchNorm stats.
        self.text_encoder.eval()
        self.semantics_encoder.eval()
        return self

    def load_vision_cache(self, f2raw_array):
        """`f2raw_array`: (N, 2048) array, memory-mapped or in memory."""
        self._f2raw_cache = f2raw_array

    def load_text_cache(self, bert_static_array):
        """`bert_static_array`: (K, 768) pooled BERT output, one row per class."""
        self._bert_static_cache = torch.from_numpy(bert_static_array).float()

    def get_vision_features(self, images, idx=None):
        f1 = self.detail_encoder(images)

        if idx is not None and self._f2raw_cache is not None:
            idx_np = idx.detach().cpu().numpy() if torch.is_tensor(idx) else idx
            f2_raw = torch.from_numpy(self._f2raw_cache[idx_np].copy()).to(
                device=images.device, dtype=images.dtype
            )
        else:
            f2_raw = self.semantics_encoder(images)

        f2 = self.semantics_proj(f2_raw)
        f2_star = self.alpha * self.visual_adapter(f2) + (1 - self.alpha) * f2

        v = torch.cat([f1, f2_star], dim=-1)
        return F.normalize(v, p=2, dim=-1)

    def get_text_features(self, input_ids=None, attention_mask=None):
        """
        Returns (K, 1024), one embedding per class, from the static hard
        prompt. This never varies per sample: training and evaluation both
        compare against the same K rows, so they are the same computation.
        """
        if self._bert_static_cache is not None:
            pooled = self._bert_static_cache.to(
                device=self.text_proj.weight.device, dtype=self.text_proj.weight.dtype
            )
        else:
            with torch.no_grad():
                pooled = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).pooler_output
        t = self.text_proj(pooled)
        return F.normalize(t, p=2, dim=-1)

    def forward(self, images, idx=None, input_ids=None, attention_mask=None):
        v_e = self.get_vision_features(images, idx=idx)  # (B, 1024)
        t_e = self.get_text_features(input_ids, attention_mask)  # (K, 1024)

        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100))
        scale = self.logit_scale.exp()

        logits = scale * v_e @ t_e.t()  # (B, K)
        return logits, v_e, t_e, scale


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    model = TrafficCLIP()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total}")
    logging.info(f"Trainable parameters: {trainable} ({trainable / total:.2%})")
