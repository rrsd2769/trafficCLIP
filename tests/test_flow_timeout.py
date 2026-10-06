"""
Idle-timeout flow segmentation, and dropping all-zero images.

`flow_timeout` defaults to 0, i.e. disabled, so the build reproduces the
original behaviour byte for byte. `drop_empty` defaults on, since an
all-black image carries no signal.
"""

import numpy as np
import pytest
from scapy.all import IP, TCP, Ether, Raw

from src.make_traffic_dataset import (
    FlowAccumulator,
    flow_key,
    process_class_folder,
)


def _pkt(ts, payload=b"p" * 80, sport=1000, dport=80):
    p = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=sport, dport=dport) / Raw(payload)
    p.time = ts
    return p


def _write(tmp_path, name, packets):
    from scapy.utils import wrpcap

    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    wrpcap(str(folder / f"{name}.pcap"), packets)
    return folder


def test_timeout_disabled_merges_distant_packets(tmp_path):
    """
    The original build's behaviour: two bursts a day apart share a 5-tuple and are
    merged into one 'flow', so mean_IAT reflects the capture duration rather
    than any traffic behaviour.
    """
    day = 86400.0
    packets = [_pkt(0.0), _pkt(0.1), _pkt(day), _pkt(day + 0.1)]
    folder = _write(tmp_path, "Demo", packets)

    imgs, _, meta = process_class_folder(folder, 0, 10, flow_timeout=0.0)
    assert len(imgs) == 1, "without a timeout the bursts must stay one flow"
    mean_iat_ms = meta[0][0]
    assert mean_iat_ms > 1e7, f"expected a capture-duration IAT, got {mean_iat_ms}"


def test_timeout_splits_distant_bursts(tmp_path):
    day = 86400.0
    packets = [_pkt(0.0), _pkt(0.1), _pkt(day), _pkt(day + 0.1)]
    folder = _write(tmp_path, "Demo", packets)

    imgs, _, meta = process_class_folder(folder, 0, 10, flow_timeout=60.0)
    assert len(imgs) == 2, "a 60s idle timeout must split the two bursts"
    for mean_iat_ms, jitter_ms, _ in meta:
        assert mean_iat_ms == pytest.approx(100.0, abs=1.0)
        assert jitter_ms == pytest.approx(0.0, abs=1.0)


def test_gap_exactly_at_the_timeout_does_not_split(tmp_path):
    """Boundary: the split triggers on gap > timeout, not >=."""
    folder = _write(tmp_path, "Demo", [_pkt(0.0), _pkt(60.0)])
    imgs, _, _ = process_class_folder(folder, 0, 10, flow_timeout=60.0)
    assert len(imgs) == 1


def test_timeout_keeps_distinct_conversations_apart(tmp_path):
    """Segmentation is per flow key; unrelated flows are unaffected."""
    packets = [
        _pkt(0.0, sport=1000, dport=80),
        _pkt(0.1, sport=2000, dport=443),
        _pkt(0.2, sport=1000, dport=80),
    ]
    folder = _write(tmp_path, "Demo", packets)
    imgs, _, _ = process_class_folder(folder, 0, 10, flow_timeout=60.0)
    assert len(imgs) == 2


def test_empty_images_are_dropped_by_default(tmp_path):
    """
    A single header-only packet (no payload) is entirely inside the 54-byte
    mask, so its image is all zeros and carries no signal.
    """
    header_only = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1, dport=2)
    header_only.time = 0.0
    real = _pkt(1.0, sport=3, dport=4)

    folder = _write(tmp_path, "Demo", [header_only, real])

    kept, _, _ = process_class_folder(folder, 0, 10, drop_empty=True)
    assert len(kept) == 1
    assert kept[0].any()

    all_flows, _, _ = process_class_folder(folder, 0, 10, drop_empty=False)
    assert len(all_flows) == 2
    assert not all_flows[0].any(), "the header-only flow should be an all-zero image"


def test_dropping_empties_still_reaches_the_sample_target(tmp_path):
    """Dropped flows are replaced by the next available one, not left short."""
    packets = []
    for i in range(5):
        h = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=100 + i, dport=1)
        h.time = float(i)
        packets.append(h)
    for i in range(3):
        packets.append(_pkt(10.0 + i, sport=200 + i, dport=2))

    folder = _write(tmp_path, "Demo", packets)
    imgs, _, _ = process_class_folder(folder, 0, 3, drop_empty=True)
    assert len(imgs) == 3
    assert all(im.any() for im in imgs)
