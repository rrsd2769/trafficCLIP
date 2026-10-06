"""
Builds a fourth independent dataset (IoT-23, Garcia et al./Stratosphere Lab,
Zenodo DOI 10.5281/zenodo.4743746) using `src/make_traffic_dataset.py`'s own
`process_pcaps_to_numpy`, so Tracks A, A-Full, and C can be re-run on it
unmodified via A.dataset.stratified_splits.

Data layout, and why this is a small, hand-picked subset
------------------------------------------------------------
The full IoT-23 corpus is ~21GB across 23 scenario captures, several of which
run to gigabytes on their own (some malicious captures ran unattended for
24+ hours). Per the guide's 2-3GB budget, this build uses 9 of the 23
scenarios, chosen to total ~2.5GB while still covering both classes IoT-23
splits into (real, unmodified device traffic vs. real malware infection
traffic) with more than one capture per malware family where available:

  Benign (3 real IoT devices, not simulated):
    benign_AmazonEcho     -- CTU-Honeypot-Capture-5-1
    benign_PhilipsHue     -- CTU-Honeypot-Capture-4-1 (shipped .pcap.xz;
                              decompressed on first run, see `ensure_decompressed`)
    benign_SomfyDoorlock  -- CTU-Honeypot-Capture-7-1 (6 short sub-captures)

  Malware (6 families/scenarios):
    malware_Mirai         -- CTU-IoT-Malware-Capture-44-1 (1.8GB; long-running
                              DDoS/scan floods, why flow_timeout matters here too)
    malware_HideAndSeek   -- CTU-IoT-Malware-Capture-1-1
    malware_Muhstik       -- CTU-IoT-Malware-Capture-3-1
    malware_Torii         -- CTU-IoT-Malware-Capture-20-1 + -21-1 (both Torii)
    malware_Trojan        -- CTU-IoT-Malware-Capture-42-1
    malware_Hakai         -- CTU-IoT-Malware-Capture-8-1

Each raw_iot23_downloads/<label>/ folder IS the class (via get_class_folders,
same convention `src/make_traffic_dataset.py` uses for USTC-TFC2016), so no
filename parsing is needed, unlike the 5G/ISCXVPN2016 builds.

flow_timeout=60.0 (the same "t60" 60-second idle-timeout setting already used
for `ustc_10class_t60.npz`/`ustc_20class_t60.npz`) is used here for the same
reason `extract_5g_dataset.py` needs it: Mirai's capture in particular is a
long-running C&C/DDoS session that would otherwise collapse mostly into a
handful of giant same-5-tuple "flows" instead of many flow-sized samples.

Classes are not balanced to a fixed samples_per_class; the smallest captures
(Torii, Trojan, Hakai run only a few MB) will naturally yield far fewer flows
than Mirai or Amazon Echo. Ceiling only, reported honestly in the build log.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/extract_iot23_dataset.py
"""

import logging
import lzma
import shutil
from pathlib import Path

from src.make_traffic_dataset import process_pcaps_to_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DIR = Path("data/raw_iot23_downloads")
OUT_PATH = Path("data/processed/iot23_9class.npz")
CAP_PER_CLASS = 900  # ceiling, not a target; see module docstring
FLOW_TIMEOUT = 60.0  # seconds; see module docstring for why this is not 0


def ensure_decompressed(raw_dir: Path) -> None:
    """`process_pcaps_to_numpy` only globs `*.pcap`; the Philips Hue capture
    ships as `.pcap.xz`, so it must be decompressed onto disk once, in place."""
    for xz_path in raw_dir.rglob("*.pcap.xz"):
        out_path = xz_path.with_suffix("")
        if out_path.exists():
            continue
        logging.info(f"decompressing {xz_path.name} -> {out_path.name}")
        with lzma.open(xz_path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def main():
    ensure_decompressed(RAW_DIR)
    process_pcaps_to_numpy(
        RAW_DIR,
        OUT_PATH,
        samples_per_class=CAP_PER_CLASS,
        mask_mode="flow",
        flow_timeout=FLOW_TIMEOUT,
    )


if __name__ == "__main__":
    main()
