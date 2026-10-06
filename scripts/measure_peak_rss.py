"""
Peak RSS must stay under 4 GB while parsing the largest capture.

The original build called scapy.rdpcap(), which materialises every packet as
a Python object at roughly 10-20x the file size in RAM, and gets OOM-killed
on a 16 GB machine. This measures the streaming path against the biggest
file present.

    PYTHONPATH=. .venv/bin/python scripts/measure_peak_rss.py
    PYTHONPATH=. .venv/bin/python scripts/measure_peak_rss.py --pcap path/to.pcap
"""

import argparse
import logging
import resource
import sys
import time
from pathlib import Path

from src.make_traffic_dataset import FlowAccumulator, flow_key, iter_packets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LIMIT_GB = 4.0


def peak_rss_gb() -> float:
    # macOS reports ru_maxrss in bytes; Linux in kilobytes.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return peak / divisor


def largest_pcap(root: Path) -> Path | None:
    pcaps = list(root.rglob("*.pcap"))
    return max(pcaps, key=lambda p: p.stat().st_size) if pcaps else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit-gb", type=float, default=LIMIT_GB)
    args = parser.parse_args()

    pcap = args.pcap or largest_pcap(args.raw_dir)
    if pcap is None:
        logging.error(f"No .pcap files under {args.raw_dir}")
        return 1

    size_gb = pcap.stat().st_size / 1024**3
    logging.info(f"Parsing {pcap} ({size_gb:.2f} GB)")

    start = time.perf_counter()
    flows: dict[tuple, FlowAccumulator] = {}
    n_packets = 0

    for pkt in iter_packets(pcap):
        n_packets += 1
        key = flow_key(pkt)
        if key is None:
            continue
        acc = flows.get(key)
        if acc is None:
            acc = flows[key] = FlowAccumulator()
        acc.add(pkt)

        if n_packets % 200_000 == 0:
            logging.info(
                f"  {n_packets:,} packets, {len(flows):,} flows, "
                f"peak RSS {peak_rss_gb():.2f} GB"
            )

    elapsed = time.perf_counter() - start
    peak = peak_rss_gb()

    logging.info(
        f"Done: {n_packets:,} packets, {len(flows):,} flows in {elapsed:.1f}s "
        f"({n_packets / max(elapsed, 1e-9):,.0f} pkt/s)"
    )
    logging.info(f"Peak RSS: {peak:.2f} GB (limit {args.limit_gb:.1f} GB)")

    if peak > args.limit_gb:
        logging.error("GATE FAILED: peak RSS exceeded the limit")
        return 1
    logging.info("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
