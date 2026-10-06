"""
Small transformer-encoder baseline, a lightweight BERT-style byte-sequence
encoder for encrypted traffic classification, reading each flow's masked
byte sequence as a token sequence with a learned positional embedding,
rather than a 2D image or a recurrent sequence.

Input: the existing masked 784-byte flow representation in
data/processed/ustc_*.npz (`x`, shape (N,1,28,28)), flattened to (N,784,1),
each byte position treated as one token. This is a compact, from-scratch
transformer sized to train in minutes on this project's hardware budget; it
is not pretrained and makes no claim of parity with a pretrained language
model such as BERT or ET-BERT.
"""

import torch
import torch.nn as nn


class ByteTransformer(nn.Module):
    def __init__(self, num_classes, seq_len=784, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        tokens = x.view(x.size(0), -1, 1)  # (B, 1, 28, 28) -> (B, 784, 1)
        tokens = self.input_proj(tokens) + self.pos_embed
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
