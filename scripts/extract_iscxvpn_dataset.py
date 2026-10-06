"""
Builds a second, independent dataset (ISCXVPN2016) for Phase 3's cross-dataset
generalization check, using the identical byte-image/stats preprocessing as
`src/make_traffic_dataset.py` builds ustc_*.npz, so Tracks A, A-Full, and C
can be re-run on it unmodified via A.dataset.stratified_splits.

Data layout, and why the class scheme differs from USTC-TFC2016
------------------------------------------------------------------
ISCXVPN2016 (Gil et al., ICISSP 2016) is not organised as one folder per
class the way USTC-TFC2016 is; it ships as individual captures per
application/session, named things like `facebook_audio1a.pcap` or
`vpn_hangouts_chat1a.pcap`. The captures staged under `data/raw_iscxvpn_downloads/`
(`NonVPN-PCAPs-01/`, `VPN-PCAPS-01/`) are the "01" split of the full corpus,
37 captures total: no browsing/streaming captures are present in this split.

Labels are assigned by parsing each filename against FILENAME_CLASS_RULES
below, using the dataset's own published traffic-type taxonomy (chat, email,
file transfer, P2P, voip/video). VPN and non-VPN captures of the same traffic
type are folded into the same class, so the task stays "classify traffic
type", the same kind of task as USTC-TFC2016 (application/family
classification) rather than "detect VPN encapsulation" (the ISCXVPN2016
paper's own, different, headline experiment). This is a documented choice,
not a claim about which framing is more standard.

Because this split has only 1-2 captures for some classes (P2P: 1, File
Transfer: 2) against Chat's 15, the classes are NOT balanced to a fixed
samples_per_class the way USTC's 900/class is; each class takes every flow
its captures yield, up to CAP_PER_CLASS as a ceiling only. The resulting
class distribution is reported, not hidden, in the build log.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/extract_iscxvpn_dataset.py
"""

import logging
import re
from pathlib import Path

import numpy as np

from src.make_traffic_dataset import FlowAccumulator, flow_key, iter_packets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DIRS = [
    Path("data/raw_iscxvpn_downloads/NonVPN-PCAPs-01"),
    Path("data/raw_iscxvpn_downloads/VPN-PCAPS-01"),
]
OUT_PATH = Path("data/processed/iscxvpn_5class.npz")
CAP_PER_CLASS = 900  # ceiling, not a target; see module docstring

# Ordered so the first match wins; checked against the lowercased filename
# stem with the "vpn_" prefix stripped. Matches ISCXVPN2016's own published
# traffic-type categories (Chat, Email, File Transfer, P2P, VoIP/Video).
FILENAME_CLASS_RULES = [
    (re.compile(r"chat"), "Chat"),
    (re.compile(r"^email"), "Email"),
    (re.compile(r"ftps|sftp|^scp"), "FileTransfer"),
    (re.compile(r"bittorrent"), "P2P"),
    (re.compile(r"audio|video|hangouts"), "VoIPVideo"),
]


def classify_filename(path: Path) -> str | None:
    stem = path.stem.lower()
    if stem.startswith("vpn_"):
        stem = stem[len("vpn_"):]
    for pattern, label in FILENAME_CLASS_RULES:
        if pattern.search(stem):
            return label
    return None


def collect_files() -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    unmatched = []
    for raw_dir in RAW_DIRS:
        for path in sorted(raw_dir.glob("*.pcap*")):
            label = classify_filename(path)
            if label is None:
                unmatched.append(path)
                continue
            by_class.setdefault(label, []).append(path)

    if unmatched:
        logging.warning(f"{len(unmatched)} files matched no class rule: {[p.name for p in unmatched]}")

    return by_class


def process_class(label: str, files: list[Path], cap: int):
    images, metadata = [], []
    n_dropped = 0
    logging.info(f"--- Class {label}: {len(files)} capture(s) ---")

    for pcap_file in files:
        if len(images) >= cap:
            break

        segments: list[FlowAccumulator] = []
        open_segment: dict[tuple, int] = {}

        for pkt in iter_packets(pcap_file):
            key = flow_key(pkt)
            if key is None:
                continue
            idx = open_segment.get(key)
            if idx is None:
                idx = open_segment[key] = len(segments)
                segments.append(FlowAccumulator(mask_mode="flow"))
            segments[idx].add(pkt)

        for acc in segments:
            if len(images) >= cap:
                break
            img = acc.image()
            if not img.any():
                n_dropped += 1
                continue
            images.append(img)
            metadata.append(acc.stats())

        logging.info(
            f"  {pcap_file.name}: {len(segments)} flows seen, "
            f"{len(images)} samples so far"
            + (f", {n_dropped} empty dropped" if n_dropped else "")
        )

    return images, metadata


def main():
    by_class = collect_files()
    class_names = sorted(by_class.keys())
    logging.info(f"Classes: {[(c, len(by_class[c])) for c in class_names]}")

    all_images, all_labels, all_metadata = [], [], []
    for label_idx, label in enumerate(class_names):
        images, metadata = process_class(label, by_class[label], CAP_PER_CLASS)
        all_images.extend(images)
        all_labels.extend([label_idx] * len(images))
        all_metadata.extend(metadata)
        logging.info(f"Class {label} ({label_idx}): {len(images)} total samples")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        x=np.array(all_images),
        y=np.array(all_labels),
        m=np.array(all_metadata),
        labels=class_names,
    )
    logging.info(f"wrote {OUT_PATH} ({len(all_labels)} samples, {len(class_names)} classes)")

    counts = {c: all_labels.count(i) for i, c in enumerate(class_names)}
    logging.info(f"Final class distribution: {counts}")


if __name__ == "__main__":
    main()
