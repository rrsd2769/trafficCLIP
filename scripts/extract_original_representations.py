"""
Builds FlowPic's and FS-Net's own published input representations from the
re-acquired raw PCAPs, instead of the 28x28 byte-flow image both baselines
were substituted onto in `C/baselines/` (see `C/README.md`'s scope note).

Walks the same class folders, in the same sorted order, with the same
`flow_key`/`iter_packets` streaming logic and the same 900-flow-per-class cap
and drop-empty criterion as `src/make_traffic_dataset.py`, so the resulting
flow order and accepted-flow set are the same set of flows already labelled
in `data/processed/ustc_20class.npz`. Each flow also runs through the
existing `FlowAccumulator` unmodified, purely to reproduce the identical
image-based drop-empty decision; the image itself is discarded here.

Two representations are derived per accepted flow:

  FlowPic: a 32x32 histogram of packet size (0-1500 bytes) against time
  since the flow's first packet (0-60s, clipped), log1p-scaled and
  normalised to [0, 1]. The published FlowPic bins into a much larger grid
  before downsampling; the released code was never open-sourced, so this is
  a documented reproduction of the paper's description (Section III-B),
  not a verified match to unreleased binning code.

  FS-Net: each packet's length, signed by direction (+ if the packet's
  (src_ip, src_port) is the flow key's first canonical endpoint, else -),
  truncated or zero-padded to MAX_SEQ_LEN packets. The published paper does
  not fix a single max length across datasets; MAX_SEQ_LEN is this
  reproduction's own choice, documented rather than presented as the
  original figure.

Output written to data/processed/ustc_20class_{flowpic,fsnet}_orig.npz,
then sliced to the 10-class subset with the existing `derive_subset`.
`C/tests/test_original_repr_alignment.py` gates every downstream number on
the y and m arrays of both new archives exactly matching
ustc_20class.npz's, i.e. that this walk landed on the identical flows.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/extract_original_representations.py
"""

import logging
from pathlib import Path

import numpy as np
import scapy.all as scapy

from src.make_traffic_dataset import (
    FlowAccumulator,
    derive_subset,
    flow_key,
    get_class_folders,
    iter_packets,
)
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FLOWPIC_SIZE = 32
FLOWPIC_MAX_TIME_S = 60.0
FLOWPIC_MAX_PKT_BYTES = 1500.0
FSNET_MAX_SEQ_LEN = 64


class OriginalReprFlow:
    """
    Wraps the existing FlowAccumulator (for the identical drop-empty gate)
    and separately records what FlowPic's and FS-Net's own representations
    need: a (size, elapsed_time) pair per packet for FlowPic, and a signed
    packet length per packet for FS-Net.
    """

    __slots__ = ("image_acc", "first_endpoint", "size_time", "signed_lengths", "_t0")

    def __init__(self, first_endpoint):
        self.image_acc = FlowAccumulator(mask_mode="flow")
        self.first_endpoint = first_endpoint
        self.size_time = []
        self.signed_lengths = []
        self._t0 = None

    def add(self, pkt):
        self.image_acc.add(pkt)

        raw_len = len(scapy.raw(pkt))
        ts = float(pkt.time)
        if self._t0 is None:
            self._t0 = ts
        self.size_time.append((raw_len, ts - self._t0))

        ip = pkt[scapy.IP]
        sport = pkt[scapy.TCP].sport if pkt.haslayer(scapy.TCP) else (
            pkt[scapy.UDP].sport if pkt.haslayer(scapy.UDP) else 0
        )
        sign = 1.0 if (ip.src, sport) == self.first_endpoint else -1.0
        self.signed_lengths.append(sign * raw_len)

    def is_empty(self) -> bool:
        return not self.image_acc.image().any()

    def flowpic_histogram(self) -> np.ndarray:
        sizes = np.array([s for s, _ in self.size_time], dtype=np.float64)
        times = np.array([t for _, t in self.size_time], dtype=np.float64)
        sizes = np.clip(sizes, 0, FLOWPIC_MAX_PKT_BYTES)
        times = np.clip(times, 0, FLOWPIC_MAX_TIME_S)

        hist, _, _ = np.histogram2d(
            times,
            sizes,
            bins=FLOWPIC_SIZE,
            range=[[0, FLOWPIC_MAX_TIME_S], [0, FLOWPIC_MAX_PKT_BYTES]],
        )
        hist = np.log1p(hist)
        if hist.max() > 0:
            hist = hist / hist.max()
        return hist.astype(np.float32).reshape(1, FLOWPIC_SIZE, FLOWPIC_SIZE)

    def fsnet_sequence(self) -> np.ndarray:
        seq = np.array(self.signed_lengths[:FSNET_MAX_SEQ_LEN], dtype=np.float32)
        if len(seq) < FSNET_MAX_SEQ_LEN:
            seq = np.pad(seq, (0, FSNET_MAX_SEQ_LEN - len(seq)))
        # Scale to roughly [-1, 1]; raw byte counts otherwise dominate the
        # GRU's input scale.
        return seq / FLOWPIC_MAX_PKT_BYTES


def process_class_folder(folder: Path, label: int, samples_per_class: int):
    flowpic_imgs, fsnet_seqs, labels, metadata = [], [], [], []
    n_dropped = 0
    logging.info(f"Processing Class: {folder.name}")

    for pcap_file in sorted(folder.rglob("*.pcap")):
        if len(labels) >= samples_per_class:
            break

        segments: list[OriginalReprFlow] = []
        open_segment: dict[tuple, int] = {}

        for pkt in iter_packets(pcap_file):
            key = flow_key(pkt)
            if key is None:
                continue
            idx = open_segment.get(key)
            if idx is None:
                idx = open_segment[key] = len(segments)
                first_endpoint = key[0]  # sorted (ip, port) pair, smaller first
                segments.append(OriginalReprFlow(first_endpoint))
            segments[idx].add(pkt)

        for seg in segments:
            if len(labels) >= samples_per_class:
                break
            if seg.is_empty():
                n_dropped += 1
                continue
            flowpic_imgs.append(seg.flowpic_histogram())
            fsnet_seqs.append(seg.fsnet_sequence())
            labels.append(label)
            metadata.append(seg.image_acc.stats())

        logging.info(
            f"  {pcap_file.name}: {len(segments)} flows seen, "
            f"{len(labels)}/{samples_per_class} samples collected"
            + (f", {n_dropped} empty dropped" if n_dropped else "")
        )

    if len(labels) < samples_per_class:
        logging.warning(
            f"Class {folder.name}: only {len(labels)} flows available, "
            f"{samples_per_class} requested"
        )

    return flowpic_imgs, fsnet_seqs, labels, metadata


def build(data_dir: Path, samples_per_class: int, out_dir: Path):
    class_folders, _ = get_class_folders(data_dir)
    label_map = {folder.name: i for i, folder in enumerate(class_folders)}
    logging.info(f"Building original-representation archives with label map: {label_map}")

    all_flowpic, all_fsnet, all_labels, all_metadata = [], [], [], []
    for folder in class_folders:
        label = label_map[folder.name]
        fp, fs, lbl, meta = process_class_folder(folder, label, samples_per_class)
        all_flowpic.extend(fp)
        all_fsnet.extend(fs)
        all_labels.extend(lbl)
        all_metadata.extend(meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(label_map.keys())

    flowpic_path = out_dir / "ustc_20class_flowpic_orig.npz"
    np.savez_compressed(
        flowpic_path,
        x=np.array(all_flowpic),
        y=np.array(all_labels),
        m=np.array(all_metadata),
        labels=class_names,
    )
    logging.info(f"wrote {flowpic_path} ({len(all_labels)} samples)")

    fsnet_path = out_dir / "ustc_20class_fsnet_orig.npz"
    np.savez_compressed(
        fsnet_path,
        x=np.array(all_fsnet),
        y=np.array(all_labels),
        m=np.array(all_metadata),
        labels=class_names,
    )
    logging.info(f"wrote {fsnet_path} ({len(all_labels)} samples)")

    return flowpic_path, fsnet_path


def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    samples_per_class = config["dataset"]["traffic"]["samples_per_class"]
    subset = config["dataset"]["traffic"]["subset_10class"]
    out_dir = Path("data/processed")

    flowpic_20, fsnet_20 = build(raw_dir, samples_per_class, out_dir)

    derive_subset(flowpic_20, out_dir / "ustc_10class_flowpic_orig.npz", subset)
    derive_subset(fsnet_20, out_dir / "ustc_10class_fsnet_orig.npz", subset)


if __name__ == "__main__":
    main()
