"""
1D-CNN baseline (Wang et al.-style raw-byte 1D convolutional classifier for
encrypted traffic classification), reading each flow's masked byte sequence
as a 1D signal rather than a 2D image.

Input: the existing masked 784-byte flow representation in
data/processed/ustc_*.npz (`x`, shape (N,1,28,28)), flattened to (N,1,784).
No new data acquisition is needed; this baseline exists to contextualize the
task against an architecturally unrelated family (1D convolution over raw
bytes) in addition to FlowPic's 2D image and FS-Net's recurrent sequence.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, num_classes, in_channels=1, seq_len=784):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 64, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, in_channels, seq_len)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        seq = x.view(x.size(0), 1, -1)  # (B, 1, 28, 28) -> (B, 1, 784)
        return self.classifier(self.features(seq))
