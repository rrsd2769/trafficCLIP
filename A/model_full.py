"""
TrafficCLIPFull: A/model.py's TrafficCLIP extended with a physics-stats
branch and per-sample dynamic prompts, fused through a non-linear MLP head
instead of cosine similarity. This is the tri-modal design of
src/models/opt_traffic_clip.py, implemented independently.

Four properties it holds by construction:

  - alpha is an explicit keyword argument on TrafficCLIP.__init__, with no
    positional slot ahead of it that could shadow it.
  - Training and evaluation run the same K-hypothesis computation
    (hypothesis_logits), so there is no separate evaluation path that could
    see the true label in its own text input.
  - A/dataset_full.py reuses A/dataset.py's SPLIT_SEED and split fractions,
    so the partition is fixed once per archive rather than per seed.
  - text_encoder and semantics_encoder stay pinned to .eval() by the
    inherited TrafficCLIP.train(), whatever train_full.py calls .train() on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from A.model import TrafficCLIP


class StatsProjectionHead(nn.Module):
    """Projects the 3-dim physics-stats vector (IAT, jitter, entropy) into the joint embedding space."""

    def __init__(self, stats_dim=3, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    """2-layer MLP over the concatenated [vision, text, stats] embedding -> K-way logits."""

    def __init__(self, num_classes, input_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.mlp(x)


class TrafficCLIPFull(TrafficCLIP):
    def __init__(
        self,
        num_classes,
        broken_conv1=False,
        num_detail_blocks=2,
        vision_dim=1024,
        text_dim=1024,
        stats_dim=512,
        alpha=0.9,
    ):
        super().__init__(alpha=alpha, broken_conv1=broken_conv1, num_detail_blocks=num_detail_blocks)
        self.num_classes = num_classes
        self.stats_proj = StatsProjectionHead(stats_dim=3, output_dim=stats_dim)
        self.fusion_head = FusionHead(num_classes, input_dim=vision_dim + text_dim + stats_dim)

        # Populated by load_hyp_text_cache. A sample's stats are fixed for
        # the whole run, so its K dynamic-prompt embeddings are too; caching
        # them turns each step into matrix ops instead of B*K BERT calls.
        self._bert_hyp_cache = None

    def load_hyp_text_cache(self, bert_hyp_array):
        """`bert_hyp_array`: (N, K, 768) memory-mapped array."""
        self._bert_hyp_cache = bert_hyp_array

    def get_dynamic_text_features(self, input_ids, attention_mask):
        """
        input_ids/attention_mask: (M, L) flattened batch of dynamic-prompt
        token sequences; the caller reshapes the (M, 1024) result. Computed
        live, since per-sample dynamic text has no fixed K-row bank to
        precompute the way the static prompts do.
        """
        with torch.no_grad():
            pooled = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        t = self.text_proj(pooled)
        return F.normalize(t, p=2, dim=-1)

    def hypothesis_logits(self, images, stats, input_ids=None, attention_mask=None, idx=None):
        """
        K-hypothesis forward. For each of the B images, scores all K
        (candidate class name, this sample's own stats) prompts through the
        fusion head and returns ([B, K] logits, [B, 1024] v_e). Hypothesis
        k's score is the fusion head's class-k logit when told "this is
        class k", i.e. the diagonal of the [B, K, K] output. argmax over k
        is the prediction, and cross-entropy against the true label is the
        training objective in A/train_full.py as well as the eval metric.
        v_e comes back too so the caller's contrastive loss does not need a
        second forward through the detail encoder.

        With a hyp-text cache loaded and `idx` given, the K embeddings are
        read from it and input_ids/attention_mask are unused. Without one,
        they must be the (B*K, L) tokenized prompts.
        """
        b = images.size(0)
        k = self.num_classes

        v_e = self.get_vision_features(images, idx=idx)  # [B, 1024]
        s_e = F.normalize(self.stats_proj(stats), p=2, dim=-1)  # [B, 512]

        if idx is not None and self._bert_hyp_cache is not None:
            idx_np = idx.detach().cpu().numpy() if torch.is_tensor(idx) else idx
            pooled = torch.from_numpy(self._bert_hyp_cache[idx_np].copy()).to(
                device=self.text_proj.weight.device, dtype=self.text_proj.weight.dtype
            )  # [B, K, 768]
            t_e = F.normalize(self.text_proj(pooled), p=2, dim=-1)  # [B, K, 1024]
        else:
            t_e = self.get_dynamic_text_features(input_ids, attention_mask)  # [B*K, 1024]
            t_e = t_e.view(b, k, -1)  # [B, K, 1024]

        v_e_hyp = v_e.unsqueeze(1).expand(-1, k, -1)  # [B, K, 1024]
        s_e_hyp = s_e.unsqueeze(1).expand(-1, k, -1)  # [B, K, 512]
        combined = torch.cat((v_e_hyp, t_e, s_e_hyp), dim=-1)  # [B, K, 2560]

        logits_hyp = self.fusion_head(combined)  # [B, K, K]
        return torch.diagonal(logits_hyp, dim1=1, dim2=2), v_e  # [B, K], [B, 1024]


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    model = TrafficCLIPFull(num_classes=20)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total}")
    logging.info(f"Trainable parameters: {trainable} ({trainable / total:.2%})")
