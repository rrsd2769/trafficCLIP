"""
The streaming `FlowAccumulator` must produce byte-identical images and
identical statistics to the original whole-file `flow_to_image_and_stats`.

The streaming reader exists to keep peak RSS bounded on a 16 GB machine, and
is only worth having if it changes nothing about the data.
"""

import numpy as np
import pytest
from scapy.all import IP, TCP, Ether, Raw

from src.make_traffic_dataset import (
    HEADER_MASK_BYTES,
    IMAGE_BYTES,
    FlowAccumulator,
    flow_to_image_and_stats,
)


def _flow(n_packets, payload_len, seed=0):
    rng = np.random.default_rng(seed)
    packets = []
    t = 1_700_000_000.0
    for i in range(n_packets):
        payload = rng.integers(0, 256, payload_len, dtype=np.uint8).tobytes()
        p = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1000, dport=80) / Raw(payload)
        t += 0.001 * (i + 1)  # non-uniform, so jitter is non-zero
        p.time = t
        packets.append(p)
    return packets


@pytest.mark.parametrize(
    "n_packets,payload_len",
    [
        (1, 40),      # single packet: IAT and jitter must be 0.0, not NaN
        (3, 20),      # short flow, well under 784 bytes
        (5, 200),     # spans the 784-byte image boundary
        (40, 500),    # long flow, image saturated early
    ],
)
def test_streaming_matches_reference(n_packets, payload_len):
    packets = _flow(n_packets, payload_len, seed=n_packets)

    ref_img, ref_stats = flow_to_image_and_stats(packets)

    acc = FlowAccumulator(mask_mode="flow")
    for p in packets:
        acc.add(p)

    np.testing.assert_array_equal(acc.image(), ref_img)
    np.testing.assert_allclose(acc.stats(), ref_stats, rtol=0, atol=1e-9)


def test_single_packet_flow_yields_zeros_not_nan():
    acc = FlowAccumulator()
    for p in _flow(1, 60):
        acc.add(p)
    mean_iat, jitter, entropy = acc.stats()
    assert mean_iat == 0.0 and jitter == 0.0
    assert np.isfinite(entropy)


def test_flow_mask_leaves_later_packet_headers_intact():
    """
    Documents the original build's behaviour, which stays the default: only the
    first 54 bytes of the *concatenated* flow are zeroed.
    """
    packets = _flow(5, 200, seed=7)
    acc = FlowAccumulator(mask_mode="flow")
    for p in packets:
        acc.add(p)
    flat = (acc.image().reshape(-1) * 255).astype(np.uint8)

    assert np.all(flat[:HEADER_MASK_BYTES] == 0)
    # The second packet's header sits just past the first packet's length and is
    # NOT zeroed under flow-level masking.
    first_len = len(bytes(packets[0]))
    assert np.any(flat[first_len : first_len + HEADER_MASK_BYTES] != 0)


def test_packet_mask_zeroes_every_packet_header():
    packets = _flow(5, 200, seed=7)
    acc = FlowAccumulator(mask_mode="packet")
    for p in packets:
        acc.add(p)
    flat = (acc.image().reshape(-1) * 255).astype(np.uint8)

    offset = 0
    for p in packets:
        if offset >= IMAGE_BYTES:
            break
        end = min(offset + HEADER_MASK_BYTES, IMAGE_BYTES)
        assert np.all(flat[offset:end] == 0), f"header at offset {offset} not masked"
        offset += len(bytes(p))


def test_masking_does_not_change_statistics():
    """Stats are computed over unmasked bytes, matching the original ordering."""
    packets = _flow(6, 150, seed=3)
    stats = []
    for mode in ("flow", "packet", "none"):
        acc = FlowAccumulator(mask_mode=mode)
        for p in packets:
            acc.add(p)
        stats.append(acc.stats())
    np.testing.assert_allclose(stats[0], stats[1], atol=0)
    np.testing.assert_allclose(stats[0], stats[2], atol=0)


def test_invalid_mask_mode_rejected():
    with pytest.raises(ValueError, match="mask_mode"):
        FlowAccumulator(mask_mode="header")
