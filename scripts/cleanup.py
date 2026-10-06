"""
Disk hygiene: remove large intermediates once their derived artifact exists.
Raw captures are deleted once the .npz they produced has been verified.

Nothing here touches source, results/, processed .npz archives or trained
checkpoints.

    python scripts/cleanup.py --raw          # delete data/raw (needs the npz)
    python scripts/cleanup.py --pycache
    python scripts/cleanup.py --all --dry-run
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_ARCHIVES = [
    ROOT / "data" / "processed" / "ustc_10class.npz",
    ROOT / "data" / "processed" / "ustc_20class.npz",
]


def human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def archives_are_usable() -> bool:
    """
    Guards the raw-data sweep: every expected archive must exist, load, and hold
    a non-empty, self-consistent set of arrays. Raw captures are only redundant
    once something real was built from them.
    """
    for archive in REQUIRED_ARCHIVES:
        if not archive.exists():
            logging.error(f"Refusing to delete raw data: {archive} does not exist")
            return False
        try:
            with np.load(archive, allow_pickle=True) as d:
                x, y, m, labels = d["x"], d["y"], d["m"], d["labels"]
        except Exception as e:
            logging.error(f"Refusing to delete raw data: {archive} will not load ({e})")
            return False

        if len(x) == 0 or len(x) != len(y) or len(x) != len(m):
            logging.error(f"Refusing to delete raw data: {archive.name} is inconsistent")
            return False
        if not np.isfinite(m).all():
            logging.error(f"Refusing to delete raw data: {archive.name} has non-finite stats")
            return False

        logging.info(f"verified {archive.name}: {len(x)} samples, {len(labels)} classes")

    return True


def sweep_raw(dry_run: bool) -> int:
    raw = ROOT / "data" / "raw"
    if not raw.exists():
        logging.info("data/raw already absent")
        return 0
    if not archives_are_usable():
        return 0

    size = dir_size(raw)
    logging.info(
        f"{'[dry-run] would remove' if dry_run else 'removing'} data/raw ({human(size)}) "
        "-- re-downloadable with scripts/01_download_pcap_data.py (~387 MB)"
    )
    if not dry_run:
        shutil.rmtree(raw)
    return size


def sweep_pycache(dry_run: bool) -> int:
    total = 0
    for cache in sorted(ROOT.rglob("__pycache__")):
        if ".venv" in cache.parts:
            continue
        size = dir_size(cache)
        total += size
        logging.info(
            f"{'[dry-run] would remove' if dry_run else 'removing'} "
            f"{cache.relative_to(ROOT)} ({human(size)})"
        )
        if not dry_run:
            shutil.rmtree(cache)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="store_true", help="Delete data/raw once the .npz archives verify.")
    parser.add_argument("--pycache", action="store_true", help="Delete __pycache__ directories.")
    parser.add_argument("--all", action="store_true", help="Run every sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Report without deleting.")
    args = parser.parse_args()

    if not (args.raw or args.pycache or args.all):
        parser.error("choose at least one sweep: --raw, --pycache, or --all")

    freed = 0
    if args.raw or args.all:
        freed += sweep_raw(args.dry_run)
    if args.pycache or args.all:
        freed += sweep_pycache(args.dry_run)

    verb = "would free" if args.dry_run else "freed"
    logging.info(f"{verb} {human(freed)}")


if __name__ == "__main__":
    sys.exit(main())
