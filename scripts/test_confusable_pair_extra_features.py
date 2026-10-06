"""
Phase 4.2: tests whether features beyond the existing 3-value physics vector
(mean IAT, jitter, byte entropy) can separate Virut from Neris, the
confusable pair Phase 4.1 found accounts for 75.8% of A-Full's residual
20-class errors (see paper/review2_findings.md, "Anatomy of a confusable
pair").

This is a feasibility probe, not a full tri-modal retrain: retraining
A-Full with a new stats branch is a multi-hour, multi-seed effort; this
script instead asks the narrower, cheaper question directly -- do these
extra features, computed from the same raw PCAPs, separate the two classes
better than the existing 3 stats do, on their own, with a simple classifier?
A negative result here would not prove a retrain is pointless, but a
positive result is a real signal it is worth doing; that framing is
reported alongside the numbers rather than left implicit.

Walks only the Virut and Neris raw-PCAP folders, with the identical
flow_key/streaming/drop-empty logic as src/make_traffic_dataset.py, so the
900 flows collected per class here are the same 900 flows (same order) as
in ustc_20class.npz's Virut/Neris rows. `main()` asserts the collected count
for each class is exactly 900 (ustc_20class.npz's own per-class cap) before
proceeding, as the alignment check; it does not re-verify the `m` array the
way scripts/extract_original_representations.py's pytest gate does, since
this script emits a different feature set, not another copy of the existing
3-value physics vector.

New features per flow, computed from the full packet sequence (not just the
first 784 bytes):
  - packet-size mean, std, skewness
  - burst count and mean burst size, at a 1s idle-gap threshold
  - byte entropy over the *entire* flow's bytes (vs. the existing stat's
    entropy, which is also over the entire flow -- extended here to also
    report entropy of only the first 200 bytes and only the last 200 bytes,
    to check whether entropy *location* rather than magnitude disambiguates
    the pair)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/test_confusable_pair_extra_features.py
"""

import logging
from pathlib import Path

import numpy as np
import scapy.all as scapy
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.make_traffic_dataset import FlowAccumulator, flow_key, iter_packets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DIR = Path("data/raw")
CLASSES = ["Neris", "Virut"]  # sorted, matches ustc_20class.npz's label order
CAP_PER_CLASS = 900
BURST_GAP_S = 1.0
OUT_CSV = Path("results/comparison/confusable_pair_extra_features.csv")


def entropy_of(raw_bytes: bytes) -> float:
    if not raw_bytes:
        return 0.0
    counts = np.bincount(np.frombuffer(raw_bytes, dtype=np.uint8), minlength=256)
    probs = counts / len(raw_bytes)
    return float(scipy.stats.entropy(probs, base=2))


class ExtraFeatureFlow:
    __slots__ = ("image_acc", "sizes", "timestamps", "head_bytes", "tail_bytes", "all_bytes_len")

    def __init__(self):
        self.image_acc = FlowAccumulator(mask_mode="flow")
        self.sizes = []
        self.timestamps = []
        self.head_bytes = bytearray()
        self.tail_bytes = bytearray()
        self.all_bytes_len = 0

    def add(self, pkt):
        self.image_acc.add(pkt)
        raw = scapy.raw(pkt)
        self.sizes.append(len(raw))
        self.timestamps.append(float(pkt.time))
        if len(self.head_bytes) < 200:
            self.head_bytes.extend(raw[: 200 - len(self.head_bytes)])
        self.tail_bytes.extend(raw)
        if len(self.tail_bytes) > 200:
            self.tail_bytes = self.tail_bytes[-200:]
        self.all_bytes_len += len(raw)

    def is_empty(self) -> bool:
        return not self.image_acc.image().any()

    def features(self) -> list[float]:
        sizes = np.array(self.sizes, dtype=np.float64)
        ts = np.array(self.timestamps, dtype=np.float64)

        size_mean = float(sizes.mean())
        size_std = float(sizes.std())
        # scipy.stats.skew divides by std**3; a near-constant packet-size
        # sequence (std ~ 0, common for short keepalive-only flows) makes
        # this NaN rather than undefined-but-real, so it is treated as 0
        # (no skew) rather than fed to the classifier as a missing value.
        raw_skew = float(scipy.stats.skew(sizes)) if len(sizes) > 2 else 0.0
        size_skew = raw_skew if np.isfinite(raw_skew) else 0.0

        bursts, current = [], [ts[0]]
        for t in ts[1:]:
            if t - current[-1] > BURST_GAP_S:
                bursts.append(len(current))
                current = [t]
            else:
                current.append(t)
        bursts.append(len(current))
        burst_count = float(len(bursts))
        burst_mean_size = float(np.mean(bursts))

        head_entropy = entropy_of(bytes(self.head_bytes))
        tail_entropy = entropy_of(bytes(self.tail_bytes))

        return [
            size_mean, size_std, size_skew,
            burst_count, burst_mean_size,
            head_entropy, tail_entropy,
        ]


FEATURE_NAMES = [
    "pkt_size_mean", "pkt_size_std", "pkt_size_skew",
    "burst_count", "burst_mean_size",
    "head200_entropy", "tail200_entropy",
]


def process_class(folder: Path, cap: int):
    flows = []
    for pcap_file in sorted(folder.rglob("*.pcap")):
        if len(flows) >= cap:
            break
        segments: list[ExtraFeatureFlow] = []
        open_segment: dict[tuple, int] = {}
        for pkt in iter_packets(pcap_file):
            key = flow_key(pkt)
            if key is None:
                continue
            idx = open_segment.get(key)
            if idx is None:
                idx = open_segment[key] = len(segments)
                segments.append(ExtraFeatureFlow())
            segments[idx].add(pkt)

        for seg in segments:
            if len(flows) >= cap:
                break
            if seg.is_empty():
                continue
            flows.append(seg.features())
        logging.info(f"  {pcap_file.name}: {len(flows)}/{cap} samples so far")
    return flows


def main():
    all_features, all_labels = [], []
    for label_idx, cls in enumerate(CLASSES):
        logging.info(f"--- {cls} ---")
        feats = process_class(RAW_DIR / cls, CAP_PER_CLASS)
        assert len(feats) == CAP_PER_CLASS, (
            f"{cls}: collected {len(feats)} flows, expected {CAP_PER_CLASS} to match "
            f"ustc_20class.npz's per-class cap -- the raw-PCAP walk has diverged from "
            f"src/make_traffic_dataset.py's, so these features would not align with "
            f"the existing labels and must not be used."
        )
        all_features.extend(feats)
        all_labels.extend([label_idx] * len(feats))
        logging.info(f"{cls}: {len(feats)} flows")

    X = np.array(all_features)
    y = np.array(all_labels)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("label," + ",".join(FEATURE_NAMES) + "\n")
        for label, row in zip(y, X):
            f.write(f"{CLASSES[label]}," + ",".join(f"{v:.6f}" for v in row) + "\n")
    logging.info(f"wrote {OUT_CSV}")

    # Per-feature separation: mean of each class, and a two-sample t-test.
    for i, name in enumerate(FEATURE_NAMES):
        a = X[y == 0, i]
        b = X[y == 1, i]
        t, p = scipy.stats.ttest_ind(a, b, equal_var=False)
        logging.info(
            f"{name}: {CLASSES[0]} mean={a.mean():.2f} median={np.median(a):.2f} | "
            f"{CLASSES[1]} mean={b.mean():.2f} median={np.median(b):.2f} | t={t:.2f} p={p:.4g}"
        )

    # 5-fold CV accuracy of a plain logistic regression on the 7 new features
    # alone, as the feasibility signal: can these features tell Virut and
    # Neris apart better than a coin flip (50%, classes are balanced)?
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    logging.info(
        f"5-fold CV accuracy, extra features only, Virut vs Neris: "
        f"{scores.mean():.4f} +/- {scores.std():.4f} (chance = 0.50)"
    )


if __name__ == "__main__":
    main()
