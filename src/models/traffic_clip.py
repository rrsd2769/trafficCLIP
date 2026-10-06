import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class TrafficAdapter(nn.Module):
    """
    Modifies the pre-trained model by appending a few trainable layers
    including a down-projection, non-linear activation, and up-projection
    """

    def __init__(self, embed_dim, bottleneck_dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

    def forward(self, x):
        return self.adapter(x)


class TrafficResidualBlock(nn.Module):
    """
    Implements the residual block for the Detail-Aware Visual Encoder.
    Structure: 1x1 Conv (Reduce) -> 3x3 Conv (Spatial) -> 1x1 Conv (Restore)
    """

    def __init__(self, in_channels, mid_channels):
        super(TrafficResidualBlock, self).__init__()

        # 1. 1x1 convolution for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 2. 3x3 convolution for capturing local spatial features
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 3. 1x1 convolution to restore original dimensionality
        self.conv3 = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # The Skip Connection

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Add the original input (Residual Connection) before final ReLU
        out += identity
        return self.relu(out)


class DetailAwareEncoder(nn.Module):
    def __init__(self):
        super(DetailAwareEncoder, self).__init__()

        # Initial expansion to 64 channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        # 1 Residual Block
        self.res_block = TrafficResidualBlock(64, 16)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 512)  # Final Feature Vector f1

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.fc(x)


class TrafficCLIP(nn.Module):
    """
    Original TrafficCLIP Model with Detail and Semantics-Aware Visual Encoder,
    Traffic Visual Adapter, and BERT Text Encoder.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha  # Residual rate for adapter

        # Module 1: Detail-Aware Visual Encoder
        self.detail_encoder = DetailAwareEncoder()

        # Module 2: Semantics-Aware Visual Encoder (ResNet-50)
        logging.info("Loading pre-trained ResNet-50 for Semantics-Aware Encoder")
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.semantics_backbone = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.semantics_backbone.parameters():
            param.requires_grad = False  # Frozen backbone

        self.semantics_proj = nn.Linear(2048, 512)  # feature vector f2

        # Module 3: Traffic Visual Adapter
        self.visual_adapter = TrafficAdapter(embed_dim=512)

        # Module 4: Text Encoder (BERT) [cite: 138, 310]
        logging.info("Loading pre-trained BERT for Text Encoder")
        self.text_encoder = AutoModel.from_pretrained("google-bert/bert-base-uncased")

        for param in self.text_encoder.parameters():
            param.requires_grad = False  # Frozen text encoder [cite: 91]

        self.text_proj = nn.Linear(768, 1024)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )  # Temperature scaling

    def get_vision_features(self, images, idx=None):
        # `idx` is accepted (and ignored) here so forward()'s call site is
        # uniform whether or not CachedFeaturesMixin is mixed in -- see
        # src/models/cached_mixin.py, which overrides this to read
        # resnet_f2raw from the feature cache when idx is supplied.
        # f1: Detail-aware features
        f1 = self.detail_encoder(images)

        # f2: Semantics-aware features
        with torch.no_grad():
            f2_raw = self.semantics_backbone(images).flatten(
                1
            )  # Corrected dimension handling
        f2 = self.semantics_proj(f2_raw)

        # f2*: Adapter residual fusion
        # f2* = a * Adapter(f2) + (1-a) * f2
        f2_star = self.alpha * self.visual_adapter(f2) + (1 - self.alpha) * f2

        # V_fused: Multi-level vision fusion via concatenation
        v_f = torch.cat([f1, f2_star], dim=-1)  # Shape: [Batch, 1024]
        return F.normalize(v_f, p=2, dim=-1)  # L2 Normalization

    def get_text_features(self, input_ids, attention_mask):
        # T_feature: Language embeddings from BERT
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_f = self.text_proj(outputs.pooler_output)
        return F.normalize(t_f, p=2, dim=-1)  # L2 Normalization

    def forward(self, images, input_ids, attention_mask, idx=None):
        # Extract normalized features. Only vision is cache-addressed here:
        # `input_ids`/`attention_mask` are sometimes the fixed K-row static
        # class bank (train()'s original-model path) and sometimes a
        # per-sample prompt (optimized model, or dynamic matching) -- which
        # one depends on args.use_stats_prompts, which forward() has no
        # visibility into, so threading `idx` into get_text_features here
        # would risk reading the wrong cache tensor for the wrong shape.
        # Text stays live (cheap: BERT over <= batch_size short sequences,
        # once per call); see cached_mixin.py's module docstring.
        v_e = self.get_vision_features(images, idx=idx)
        t_e = self.get_text_features(input_ids, attention_mask)

        # Similarity Matrix via Cosine Similarity
        # Prevents scaling the logits by more than 100 (exp(4.6052) ≈ 100)
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))

        t = self.logit_scale.exp()
        # Cross-modality Representation Fusion via Cosine Similarity
        # logits = np.dot(V_e, T_e.T) * np.exp(t)
        logits = t * torch.matmul(v_e, t_e.t())

        return logits, t


if __name__ == "__main__":
    try:
        model = TrafficCLIP()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total Parameters in TrafficCLIP: {total_params}")
        logging.info(f"Trainable Parameters in TrafficCLIP: {trainable_params}")
        logging.info(f"Trainable Ratio: {trainable_params/total_params:.2f}")
    except Exception as e:
        logging.error(f"Error initializing TrafficCLIP model: {e}")
