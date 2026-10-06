"""
FS-Net baseline (Liu et al., "FS-Net: A Flow Sequence Network For Encrypted
Traffic Classification"), reimplemented as a bidirectional multi-layer GRU
encoder that compresses a flow into a dense representation, a GRU decoder
that reconstructs the input from it, and a classifier head reading the same
representation, trained jointly on classification and reconstruction loss,
matching FS-Net's compress-reconstruct-classify design.

Two input modes, both routed through the same encoder-decoder-classifier:

  seq_len=28, input_dim=28 (default): each 28x28 flow image read as a
  28-step sequence of 28-dim row vectors, a substitution for FS-Net's own
  input because the raw PCAPs needed to extract real packet-length
  sequences were not on disk; see `C/README.md`'s scope note. Each step is
  a row of the flow-byte image, not a packet.

  seq_len=64, input_dim=1: FS-Net's own per-flow sequence of packet
  lengths signed by direction, built by
  `scripts/extract_original_representations.py` from the re-acquired raw
  PCAPs (`data/processed/ustc_*_fsnet_orig.npz`). MAX_SEQ_LEN=64 is this
  reproduction's own choice; the published paper does not fix one value
  across datasets.
"""

import torch
import torch.nn as nn


class FSNet(nn.Module):
    def __init__(self, num_classes, seq_len=28, input_dim=28, hidden_dim=64, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        enc_dim = hidden_dim * 2 * num_layers
        self.dense = nn.Sequential(nn.Linear(enc_dim, hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if x.dim() == 4:
            seq = x.squeeze(1)  # (B, 1, 28, 28) -> (B, 28, 28) pseudo-sequence
        elif x.dim() == 2:
            seq = x.unsqueeze(-1)  # (B, seq_len) -> (B, seq_len, 1) real packet lengths
        else:
            seq = x
        _, h_n = self.encoder(seq)  # h_n: (num_layers*2, B, hidden_dim)
        batch_size = seq.size(0)
        h_flat = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        z = self.dense(h_flat)  # compressed flow representation
        logits = self.classifier(z)

        dec_in = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        recon = self.output_proj(dec_out)
        return logits, recon, seq
