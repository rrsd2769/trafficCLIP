import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization

from src.models.traffic_clip import TrafficCLIP
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class StatsProjectionHead(nn.Module):
    """
    Mod 1: Projects statistical features into the joint embedding space.
    Enables the model to leverage session-level statistics alongside
    """

    def __init__(self, stats_dim, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class NonLinearFusionHead(nn.Module):
    """
    Mod 2: Replaces static Cosine Similarity with a 2-layer MLP.
    Captures non-linear correlations between visual byte patterns
    and textual behavioral anchors.
    """

    def __init__(self, num_classes, input_dim=2048, hidden_dim=512):
        """
        Phase 3 Architecture:
        - LayerNorm for multimodal stability
        - Dropout (0.4) for regularization against session-level overfitting
        """
        super().__init__()
        self.mlp = nn.Sequential(
            # Layer 1: Expansion and Non-linearity
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            # Layer 2: Classification Logits
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, combined_features):
        return self.mlp(combined_features)


class OptimizedTrafficCLIP(TrafficCLIP):
    def __init__(
        self,
        num_classes,
        stats_input_dim=None,
        use_stats=True,
        vision_dim=1024,
        text_dim=1024,
        stats_dim=512,
        alpha=0.9,
    ):
        # Defect 1 fix: TrafficCLIP.__init__ takes `alpha`, not `num_classes`
        # -- the manuscript's Eq. 3 states alpha=0.9 (the class default).
        # Passing num_classes here silently set alpha=10, so every
        # "optimized" run trained with adapter fusion 10*Adapter(f2)-9*f2
        # instead of the architecture the manuscript describes. `alpha` is
        # now an explicit constructor argument (default 0.9) so the buggy
        # value can still be reproduced deliberately for the pre-fix report.
        super().__init__(alpha=alpha)

        # Stats Projection Head for Tri-modal logic
        self.use_stats = use_stats

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Fusion Head: Non-Linear MLP instead of Cosine Similarity
        if self.use_stats:
            if stats_input_dim is None:
                raise ValueError(
                    "stats_input_dim must be provided if use_stats is True"
                )
            self.stats_proj = StatsProjectionHead(stats_input_dim, stats_dim)
            fusion_input_dim = vision_dim + text_dim + stats_dim  # 2560
        else:
            self.stats_proj = None
            fusion_input_dim = vision_dim + text_dim  # 2048

        self.fusion_head = NonLinearFusionHead(
            num_classes=num_classes, input_dim=fusion_input_dim
        )

    def forward(self, images, input_ids, attention_mask, stats_vector=None, idx=None):
        """
        Modified forward pass:
        1. Extract vision features (Detail + Semantics + Adapter)
        2. Extract text features (BERT Behavioral Anchors)
        3. Concatenate and pass through MLP Fusion Head
        """

        # Quantize Inputs
        images = self.quant(images)
        if stats_vector is not None:
            stats_vector = self.quant(stats_vector)

        # Vision Modality Representation Learning -- idx addresses the
        # feature cache when a CachedOptimizedTrafficCLIP has one loaded;
        # see src/models/cached_mixin.py.
        v_e = self.get_vision_features(images, idx=idx)  # [Batch, 1024]

        # Textual Modality Representation Learning. Not cache-addressed here
        # (see TrafficCLIP.forward's note) -- stays live.
        t_e = self.get_text_features(input_ids, attention_mask)  # [Batch, 1024]

        # Mod 2 Fusion
        if self.use_stats and stats_vector is not None:
            # Module 3: Physics-Aware Features
            s_e = self.stats_proj(stats_vector)
            s_e = F.normalize(s_e, p=2, dim=-1)
            # Tri-modal Concatenation
            combined = torch.cat((v_e, t_e, s_e), dim=1)
        else:
            # Fallback to Bi-modal Concatenation
            combined = torch.cat((v_e, t_e), dim=1)

        logits = self.fusion_head(combined)

        # Prevents scaling the logits by more than 100 (exp(4.6052) ≈ 100)
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        return logits, self.logit_scale.exp()


if __name__ == "__main__":
    try:
        config = load_config()
        traffic_cfg = config["dataset"]["traffic"]["classes"]

        num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
        model = OptimizedTrafficCLIP(
            num_classes=num_classes, stats_input_dim=3, use_stats=True
        )  # IAT, Jitter, Entropy
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total Parameters in TrafficCLIP Optimized: {total_params}")
        logging.info(
            f"Trainable Parameters in TrafficCLIP Optimized: {trainable_params}"
        )
        logging.info(f"Trainable Ratio: {trainable_params/total_params:.2f}")

    except Exception as e:
        logging.error(f"Error initializing OptimizedTrafficCLIP model: {e}")
