"""
Flow keys must not collide across distinct conversations.

The original `extract_flows` built its key as

    tuple(sorted((ip.src, ip.dst))) + tuple(sorted((sport, dport)))

sorting addresses and ports as two independent sets. That loses the pairing
between an address and its port, so two genuinely different conversations can
land in the same bucket and get concatenated into one bogus "flow" image.
"""

import numpy as np
import pytest
from scapy.all import IP, TCP, UDP, Ether, Raw

from src.make_traffic_dataset import (
    FlowAccumulator,
    calculate_stats,
    extract_flows,
    flow_key,
    flow_to_image_and_stats,
)


def legacy_flow_key(pkt):
    """The pre-fix key, reproduced so the regression is explicit."""
    ip = pkt[IP]
    sport, dport = 0, 0
    if pkt.haslayer(TCP):
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif pkt.haslayer(UDP):
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    return tuple(sorted((ip.src, ip.dst))) + tuple(sorted((sport, dport))) + (ip.proto,)


def _pkt(src, dst, sport, dport, payload=b"x" * 100, ts=0.0):
    p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport) / Raw(payload)
    p.time = ts
    return p


def test_independent_sort_collides_but_paired_sort_does_not():
    """
    Two distinct conversations that the old key merges:

      A:1000 -> B:80    (a client on port 1000 talking to a web server)
      B:1000 -> A:80    (a client on the *other* host, ports swapped)

    Independent sorting reduces both to (A, B, 80, 1000).
    """
    a, b = "10.0.0.1", "10.0.0.2"
    p1 = _pkt(a, b, 1000, 80)
    p2 = _pkt(b, a, 1000, 80)

    assert legacy_flow_key(p1) == legacy_flow_key(p2), (
        "test premise broken: the legacy key was expected to collide here"
    )
    assert flow_key(p1) != flow_key(p2), "paired-endpoint key still collides"


def test_extract_flows_separates_the_colliding_pair():
    a, b = "10.0.0.1", "10.0.0.2"
    packets = [_pkt(a, b, 1000, 80, ts=0.0), _pkt(b, a, 1000, 80, ts=0.1)]
    flows = extract_flows(packets)
    assert len(flows) == 2


def test_both_directions_of_one_conversation_share_a_key():
    """The key must stay bidirectional: request and reply belong to one flow."""
    a, b = "10.0.0.1", "10.0.0.2"
    request = _pkt(a, b, 4444, 443, ts=0.0)
    reply = _pkt(b, a, 443, 4444, ts=0.1)

    assert flow_key(request) == flow_key(reply)
    assert len(extract_flows([request, reply])) == 1


def test_non_ip_packets_are_skipped():
    p = Ether() / Raw(b"not ip")
    assert flow_key(p) is None
    assert extract_flows([p]) == {}


def test_udp_and_tcp_on_same_endpoints_are_distinct_flows():
    a, b = "10.0.0.1", "10.0.0.2"
    tcp = Ether() / IP(src=a, dst=b) / TCP(sport=1000, dport=53) / Raw(b"q")
    udp = Ether() / IP(src=a, dst=b) / UDP(sport=1000, dport=53) / Raw(b"q")
    tcp.time, udp.time = 0.0, 0.1
    assert flow_key(tcp) != flow_key(udp)
