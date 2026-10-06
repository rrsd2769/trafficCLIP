"""
Builds a third independent dataset (5G traffic, UFJF NetLab's "5G Traffic
Capture Dataset: User Equipment Classification in the 5G Core", Zenodo DOI
10.5281/zenodo.15064130) using `src/make_traffic_dataset.py`'s own
`process_pcaps_to_numpy`, so Tracks A, A-Full, and C can be re-run on it
unmodified via A.dataset.stratified_splits.

Data layout, and why flow_timeout matters here
------------------------------------------------
Only `inference_data.zip` (2.3GB) was downloaded, not the 23.6GB
`training_data.zip` -- the inference set is the one that fits the guide's
2GB-class budget. It ships four single continuous PCAP captures, one per 5G
traffic type (eMBB video, URLLC live stream, mMTC UDP telemetry), not
one-folder-per-class the way USTC-TFC2016 is, so this script first symlinks
each capture into a `by_class/<ClassName>/` layout that `get_class_folders`
expects.

Critically, each capture is ONE sustained session on the SAME 5-tuple for
its entire duration (a single YouTube stream, a single UDP telemetry run),
so with the default flow_timeout=0 (no idle-based splitting, the behaviour
`extract_iscxvpn_dataset.py` relies on because its captures are already
short per-session files) almost the whole file collapses into a single
"flow" -- confirmed empirically: an earlier version of this script without
a timeout produced only 412 samples total (186/220/6 across 3 classes).
flow_timeout=60.0 (the same "t60" 60-second idle-timeout setting already
used for `ustc_10class_t60.npz`/`ustc_20class_t60.npz`) splits each capture
into many flow-sized segments instead.

mMTC has two captures (a fixed-rate UDP generator and a variable-interval
IoT-like pattern); eMBB and URLLC have one each. Classes are not balanced to
a fixed samples_per_class; each class takes every flow its captures yield,
up to CAP_PER_CLASS as a ceiling only.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/extract_5g_dataset.py
"""

import logging
from pathlib import Path

from src.make_traffic_dataset import process_pcaps_to_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DIR = Path("data/raw_5g_downloads/inference_data")
BY_CLASS_DIR = Path("data/raw_5g_downloads/by_class")
OUT_PATH = Path("data/processed/g5_3class.npz")
CAP_PER_CLASS = 900  # ceiling, not a target; see module docstring
FLOW_TIMEOUT = 60.0  # seconds; see module docstring for why this is not 0

FILENAME_TO_CLASS = {
    "youtube-1M-1080p.pcap": "eMBB",
    "naver-tv-1M.pcap": "URLLC",
    "udp-100pps.pcap": "mMTC",
    "udp-nc-traffic-1k.pcap": "mMTC",
}


def build_by_class_symlinks() -> Path:
    for class_name in set(FILENAME_TO_CLASS.values()):
        (BY_CLASS_DIR / class_name).mkdir(parents=True, exist_ok=True)

    for filename, class_name in FILENAME_TO_CLASS.items():
        src = (RAW_DIR / filename).resolve()
        dst = BY_CLASS_DIR / class_name / filename
        if not src.exists():
            raise FileNotFoundError(src)
        if not dst.exists():
            dst.symlink_to(src)

    return BY_CLASS_DIR


def main():
    by_class_dir = build_by_class_symlinks()
    process_pcaps_to_numpy(
        by_class_dir,
        OUT_PATH,
        samples_per_class=CAP_PER_CLASS,
        mask_mode="flow",
        flow_timeout=FLOW_TIMEOUT,
    )


if __name__ == "__main__":
    main()
