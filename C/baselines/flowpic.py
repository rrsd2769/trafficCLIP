"""
FlowPic baseline (Shapira & Shavitt, "FlowPic: Encrypted Internet Traffic
Classification is as Easy as Image Recognition"), reimplemented as a compact
two-conv-block CNN classifier.

Two input modes, both routed through the same architecture:

  img_size=28 (default): the existing 28x28 masked flow-byte image (`x` in
  `data/processed/ustc_*.npz`), a substitution for FlowPic's own input
  because the raw PCAPs needed to derive it were not on disk; see
  `C/README.md`'s scope note.

  img_size=32: FlowPic's own packet-size-against-time histogram, built by
  `scripts/extract_original_representations.py` from the re-acquired raw
  PCAPs (`data/processed/ustc_*_flowpic_orig.npz`).

The architecture (conv-pool twice, then dense) follows the published design
in both cases; only the pixel semantics differ.
"""

import torch
import torch.nn as nn


class FlowPicCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, img_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, in_channels, img_size, img_size)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
