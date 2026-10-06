"""
Builds the traffic dataset: PCAP captures -> 28x28 grayscale flow images plus a
3-dimensional physics vector [mean_IAT, jitter, byte_entropy] per flow.

Output schema (`.npz`, see `docs/dataset_schema.md`):
    x      (N, 1, 28, 28) float32   flow images, 54-byte header masked, /255
    y      (N,)           int64     class index
    m      (N, 3)         float64   [mean_IAT_ms, jitter_ms, byte_entropy_bits]
    labels (K,)           <U str    class names, indexed by y -- NOT per-sample

Note the `labels` shape: it is K class names indexed by label, not an (N,) array of
per-sample names. The run book (OptimizedTrafficCLIP_Guide.docx, section 3.2)
documents it as (N,); that is wrong. `dataset.py` and `benchmark.py` both depend on
the actual K-name behaviour, so the schema is documented rather than changed.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import scapy.all as scapy
import scipy
from scapy.utils import PcapReader

from src.utils.utils import load_config

IMAGE_BYTES = 784  # 28 * 28
HEADER_MASK_BYTES = 54  # Ethernet(14) + IP(20) + TCP(20)


def get_class_folders(data_dir: Path) -> tuple[list[Path], dict[str, int]]:
    """
    Identifies class folders in the given directory and creates a label mapping.

    Folders are sorted by name so the label mapping is deterministic. Without the
    sort the mapping came from `iterdir()`, i.e. filesystem order, which differs
    between machines and between rebuilds -- silently making class index 3 mean
    different things in two runs that are supposed to be comparable.
    """
    class_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    label_map = {folder.name: i for i, folder in enumerate(class_folders)}
    logging.info(f"Label Mapping: {label_map}")
    return class_folders, label_map


def read_pcap(pcap_file: Path) -> scapy.PacketList | None:
    """
    Reads a PCAP file fully into memory and returns the packets.

    Retained for notebook use only. The dataset build path uses `iter_packets`,
    which streams; `rdpcap` materialises every packet as a Python object at
    roughly 10-20x the file size in RAM and will OOM this machine on the larger
    USTC captures (extracted SMB and Weibo run to the gigabyte range).
    """
    try:
        return scapy.rdpcap(str(pcap_file))
    except Exception as e:
        logging.error(f"Failed to read {pcap_file.name}: {e}")
        return None


def iter_packets(pcap_file: Path):
    """
    Streams packets from a PCAP one at a time, so the file is never fully resident.
    """
    pcap_file = Path(pcap_file)
    try:
        reader = PcapReader(str(pcap_file))
    except Exception as e:
        logging.error(f"Failed to open {pcap_file.name}: {e}")
        return

    with reader:
        while True:
            try:
                pkt = reader.read_packet()
            except EOFError:
                return
            except Exception as e:
                # A truncated or malformed record ends the file; keep what we have.
                logging.warning(f"Stopped reading {pcap_file.name} early: {e}")
                return
            if pkt is None:
                return
            yield pkt


def flow_key(pkt) -> tuple | None:
    """
    Bidirectional 5-tuple flow key, or None if the packet has no IP layer.

    The two endpoints are sorted as *pairs*, so (ip, port) stays together. The
    previous implementation sorted addresses and ports independently:

        tuple(sorted((ip.src, ip.dst))) + tuple(sorted((sport, dport)))

    which lets genuinely different flows collide onto one key. A->B:80 from port
    1000 and B->A:1000 from port 80 both reduce to (A, B, 80, 1000) even though
    they are distinct conversations, and their packets get concatenated into a
    single bogus "flow" image.
    """
    if not pkt.haslayer(scapy.IP):
        return None

    ip = pkt[scapy.IP]
    sport, dport = 0, 0

    if pkt.haslayer(scapy.TCP):
        sport, dport = pkt[scapy.TCP].sport, pkt[scapy.TCP].dport
    elif pkt.haslayer(scapy.UDP):
        sport, dport = pkt[scapy.UDP].sport, pkt[scapy.UDP].dport

    endpoints = tuple(sorted([(ip.src, sport), (ip.dst, dport)]))
    return endpoints + (ip.proto,)


def extract_flows(packets) -> dict[tuple, list[scapy.Packet]]:
    """
    Groups packets into bidirectional flows. Kept for the notebook and for tests;
    the build path uses `FlowAccumulator` so packets are not retained.
    """
    flows: dict[tuple, list] = {}

    for pkt in packets:
        key = flow_key(pkt)
        if key is None:
            continue
        flows.setdefault(key, []).append(pkt)

    return flows


class FlowAccumulator:
    """
    Incremental, constant-memory summary of one flow.

    Holds only what `flow_to_image_and_stats` actually consumes:
      * the first 784 bytes of the concatenated flow (the image)
      * a 256-bin byte histogram and total length over *all* bytes (the entropy)
      * inter-arrival times (mean IAT and jitter)

    This is mathematically identical to concatenating every packet and slicing,
    but the packets themselves are released as soon as they are folded in, so a
    gigabyte capture costs kilobytes per flow instead of tens of gigabytes total.
    """

    __slots__ = ("head", "byte_counts", "total_bytes", "iats", "_last_ts", "mask_mode")

    def __init__(self, mask_mode: str = "flow"):
        if mask_mode not in ("flow", "packet", "none"):
            raise ValueError(f"mask_mode must be flow|packet|none, got {mask_mode!r}")
        self.mask_mode = mask_mode
        self.head = bytearray()
        self.byte_counts = np.zeros(256, dtype=np.int64)
        self.total_bytes = 0
        self.iats: list[float] = []
        self._last_ts: float | None = None

    def add(self, pkt) -> None:
        raw = scapy.raw(pkt)

        if len(self.head) < IMAGE_BYTES:
            fragment = raw
            if self.mask_mode == "packet":
                # Zero this packet's own headers before it joins the flow image.
                # Statistics below still see the unmasked bytes, matching the
                # original ordering (stats computed before masking).
                fragment = bytes(HEADER_MASK_BYTES) + raw[HEADER_MASK_BYTES:]
            self.head.extend(fragment[: IMAGE_BYTES - len(self.head)])

        if raw:
            self.byte_counts += np.bincount(
                np.frombuffer(raw, dtype=np.uint8), minlength=256
            )
            self.total_bytes += len(raw)

        ts = float(pkt.time)
        if self._last_ts is not None:
            self.iats.append(ts - self._last_ts)
        self._last_ts = ts

    def stats(self) -> list[float]:
        """Returns [mean_IAT_ms, jitter_ms, byte_entropy_bits]."""
        if self.iats:
            iats = np.asarray(self.iats)
            mean_iat = float(np.mean(iats) * 1000)
            jitter = float(np.std(iats) * 1000)
        else:
            mean_iat, jitter = 0.0, 0.0

        if self.total_bytes > 0:
            probs = self.byte_counts / self.total_bytes
            entropy = float(scipy.stats.entropy(probs, base=2))
        else:
            entropy = 0.0

        return [mean_iat, jitter, entropy]

    def image(self) -> np.ndarray:
        """Returns the (1, 28, 28) float32 flow image, masked per `mask_mode`."""
        buffer = bytearray(self.head[:IMAGE_BYTES])
        if len(buffer) < IMAGE_BYTES:
            buffer.extend(b"\x00" * (IMAGE_BYTES - len(buffer)))

        if self.mask_mode == "flow":
            # The original behaviour: only the *concatenated flow's* first 54
            # bytes are zeroed, so packets 2..n keep their headers. Kept as the
            # default so B stays comparable with the manuscript's protocol.
            for i in range(HEADER_MASK_BYTES):
                buffer[i] = 0x00

        img = np.frombuffer(bytes(buffer), dtype=np.uint8).astype(np.float32) / 255.0
        return img.reshape(1, 28, 28)


def calculate_stats(flow_pkts, raw_bytes):
    """
    Calculates physical flow characteristics: Mean IAT, Jitter, and Byte Entropy.
    Calculated before truncation/masking for maximum accuracy.

    Non-streaming reference implementation, kept so tests can assert that
    `FlowAccumulator` agrees with it exactly.
    """
    # 1. Calculate Inter-Arrival Times (IAT)
    timestamps = [float(p.time) for p in flow_pkts]
    if len(timestamps) > 1:
        iats = np.diff(timestamps)
        # Convert to milliseconds for the prompt
        mean_iat = np.mean(iats) * 1000
        jitter = np.std(iats) * 1000
    else:
        mean_iat, jitter = 0.0, 0.0

    # 2. Calculate Byte Entropy (Shannon Entropy)
    if len(raw_bytes) > 0:
        counts = np.bincount(np.frombuffer(raw_bytes, dtype=np.uint8), minlength=256)
        probs = counts / len(raw_bytes)
        entropy = scipy.stats.entropy(probs, base=2)
    else:
        entropy = 0.0

    return mean_iat, jitter, entropy


def flow_to_image_and_stats(flow_packets, size=IMAGE_BYTES):
    """
    Extracts stats and converts flow to a 54-byte masked image.

    Non-streaming reference implementation, kept for the notebook and for the
    equivalence test against `FlowAccumulator`.
    """
    # Step A: Get FULL raw bytes for stats
    raw_bytes = b"".join([scapy.raw(p) for p in flow_packets])

    # Step B: Physics-Informed Calculation
    mean_iat, jitter, entropy = calculate_stats(flow_packets, raw_bytes)

    # Step C: Prepare Image Buffer (Truncate/Pad)
    buffer = bytearray(raw_bytes[:size])
    if len(buffer) < size:
        buffer.extend(b"\x00" * (size - len(buffer)))

    # Step D: 54-Byte Header Masking (Phase 1 Fix)
    # Masking Eth(14) + IP(20) + TCP(20) to prevent hardware fingerprints
    for i in range(0, 54):
        buffer[i] = 0x00

    img = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
    return img.reshape(1, 28, 28), [mean_iat, jitter, entropy]


def process_class_folder(
    folder: Path,
    label: int,
    samples_per_class: int,
    mask_mode: str = "flow",
    flow_timeout: float = 0.0,
    drop_empty: bool = True,
):
    """
    Streams every PCAP in a class folder and returns up to `samples_per_class`
    flow samples, in order of first appearance.

    Only the bounded `FlowAccumulator` summary is retained per flow, never the
    packets, so a gigabyte capture costs well under a gigabyte of RAM.

    Args:
        flow_timeout: idle timeout in seconds. 0 disables it, which is the
            original behaviour and the default -- see the note below.
        drop_empty: skip flows whose masked image is entirely zero.
    """
    images, labels, metadata = [], [], []
    n_dropped = 0
    logging.info(f"Processing Class: {folder.name}")

    # rglob, not glob: the .7z classes (Weibo, Cridex, Geodo, SMB, ...) extract
    # into a nested directory, and plain glob silently found zero captures there.
    for pcap_file in sorted(folder.rglob("*.pcap")):
        if len(images) >= samples_per_class:
            break

        segments: list[FlowAccumulator] = []  # in creation order
        open_segment: dict[tuple, int] = {}  # flow key -> index in `segments`
        last_seen: dict[tuple, float] = {}

        for pkt in iter_packets(pcap_file):
            key = flow_key(pkt)
            if key is None:
                continue

            ts = float(pkt.time)
            idx = open_segment.get(key)
            expired = (
                idx is not None
                and flow_timeout > 0
                and (ts - last_seen[key]) > flow_timeout
            )
            if idx is None or expired:
                idx = open_segment[key] = len(segments)
                segments.append(FlowAccumulator(mask_mode=mask_mode))

            segments[idx].add(pkt)
            last_seen[key] = ts

        for acc in segments:
            if len(images) >= samples_per_class:
                break
            img = acc.image()
            if drop_empty and not img.any():
                # Every surviving byte was masked away: the flow was a single
                # header-only packet (a bare SYN/ACK). An all-black image is
                # label noise, and there are always more flows available.
                n_dropped += 1
                continue
            images.append(img)
            labels.append(label)
            metadata.append(acc.stats())

        logging.info(
            f"  {pcap_file.name}: {len(segments)} flows seen, "
            f"{len(images)}/{samples_per_class} samples collected"
            + (f", {n_dropped} empty dropped" if n_dropped else "")
        )

    if len(images) < samples_per_class:
        logging.warning(
            f"Class {folder.name}: only {len(images)} flows available, "
            f"{samples_per_class} requested"
        )

    return images, labels, metadata


def process_pcaps_to_numpy(
    data_dir: Path,
    output_file: Path,
    samples_per_class: int = 1500,
    only_classes: list[str] | None = None,
    mask_mode: str = "flow",
    flow_timeout: float = 0.0,
    drop_empty: bool = True,
):
    """
    Builds one `.npz` archive from the class folders under `data_dir`.

    Args:
        only_classes: restrict the build to these class folder names. Labels are
            assigned over the restricted, sorted set, so a 10-class archive is
            indexed 0..9 independently of the 20-class archive.
        mask_mode: "flow" zeroes the first 54 bytes of the concatenated flow (the
            original behaviour, so packets 2..n keep their headers -- the default,
            for comparability with the manuscript). "packet" zeroes each packet's
            own 54-byte header before concatenation, which is what the base paper
            describes and what A should use. "none" disables masking entirely.
    """
    all_images, all_labels, all_metadata = [], [], []
    class_folders, _ = get_class_folders(data_dir)

    if only_classes is not None:
        wanted = set(only_classes)
        missing = wanted - {f.name for f in class_folders}
        if missing:
            raise ValueError(f"Requested classes not present in {data_dir}: {sorted(missing)}")
        class_folders = [f for f in class_folders if f.name in wanted]

    label_map = {folder.name: i for i, folder in enumerate(class_folders)}
    logging.info(f"Building {output_file.name} with label map: {label_map}")

    for folder in class_folders:
        label = label_map[folder.name]
        imgs, lbls, meta = process_class_folder(
            folder,
            label,
            samples_per_class,
            mask_mode=mask_mode,
            flow_timeout=flow_timeout,
            drop_empty=drop_empty,
        )
        all_images.extend(imgs)
        all_labels.extend(lbls)
        all_metadata.extend(meta)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 'labels' holds K class names indexed by y -- see the module docstring.
    np.savez_compressed(
        output_file,
        x=np.array(all_images),
        y=np.array(all_labels),
        m=np.array(all_metadata),
        labels=list(label_map.keys()),
    )

    logging.info(
        f"Successfully saved {len(all_images)} samples to {output_file} "
        f"(mask_mode={mask_mode}, flow_timeout={flow_timeout}, drop_empty={drop_empty})"
    )
    return output_file


def derive_subset(source_file: Path, output_file: Path, classes: list[str]) -> Path:
    """
    Writes a class-subset archive by filtering an existing one.

    Per-class flow collection is independent of which other classes are in the
    build -- a class folder is walked on its own and capped at
    `samples_per_class` -- so the rows for "Skype" in the 10-class archive are
    byte-identical to its rows in the 20-class archive. Only the label index
    differs, because labels are assigned over each archive's own sorted class
    list.

    Deriving therefore gives exactly what a second full build would, without
    re-parsing 3.7 GB of captures.
    """
    with np.load(source_file, allow_pickle=True) as d:
        x, y, m, labels = d["x"], d["y"], d["m"], list(d["labels"])

    wanted = sorted(classes)
    missing = set(wanted) - set(labels)
    if missing:
        raise ValueError(f"{source_file.name} lacks classes: {sorted(missing)}")

    old_index = {name: labels.index(name) for name in wanted}
    remap = {old_index[name]: new for new, name in enumerate(wanted)}

    keep = np.isin(y, list(remap.keys()))
    y_new = np.array([remap[int(v)] for v in y[keep]], dtype=y.dtype)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        x=x[keep],
        y=y_new,
        m=m[keep],
        labels=wanted,
    )
    logging.info(
        f"Derived {output_file.name}: {keep.sum()} samples, "
        f"{len(wanted)} classes, from {source_file.name}"
    )
    return output_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["mini", "10class", "20class", "all"],
        default="all",
        help="Which archive(s) to build. 'all' builds 10class and 20class.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["flow", "packet", "none"],
        default="flow",
        help="Header masking granularity. Default 'flow' matches the original code.",
    )
    parser.add_argument(
        "--flow-timeout",
        type=float,
        default=0.0,
        help=(
            "Idle timeout in seconds; a gap larger than this starts a new flow. "
            "Default 0 (disabled) reproduces the original behaviour -- see the "
            "Defect 5 note in docs/dataset_schema.md. 60 is the conventional value."
        ),
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep flows whose masked image is entirely zero (default: drop them).",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Appended to the output filename stem, e.g. '_packetmask'.",
    )
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    subset = config["dataset"]["traffic"]["subset_10class"]

    targets = {
        "mini": (
            Path(config["paths"]["mini_output_data_file"]),
            10,
            subset,
        ),
        "10class": (
            Path(config["paths"]["ustc_10class_file"]),
            config["dataset"]["traffic"]["samples_per_class"],
            subset,
        ),
        "20class": (
            Path(config["paths"]["ustc_20class_file"]),
            config["dataset"]["traffic"]["samples_per_class"],
            None,
        ),
    }

    def named(name: str) -> Path:
        out = targets[name][0]
        return out.with_name(f"{out.stem}{args.suffix}{out.suffix}") if args.suffix else out

    if args.variant == "all":
        # Build the 20-class archive once and slice the 10-class subset out of
        # it. See derive_subset for why this is exact.
        out20, n20, _ = targets["20class"]
        process_pcaps_to_numpy(
            raw_dir,
            named("20class"),
            samples_per_class=n20,
            only_classes=None,
            mask_mode=args.mask_mode,
            flow_timeout=args.flow_timeout,
            drop_empty=not args.keep_empty,
        )
        derive_subset(named("20class"), named("10class"), subset)
        return

    name = args.variant
    _, n, classes = targets[name]
    process_pcaps_to_numpy(
        raw_dir,
        named(name),
        samples_per_class=n,
        only_classes=classes,
        mask_mode=args.mask_mode,
        flow_timeout=args.flow_timeout,
        drop_empty=not args.keep_empty,
    )


if __name__ == "__main__":
    main()
