"""
Gates every original-representation (FlowPic/FS-Net) number on the y and m
arrays of scripts/extract_original_representations.py's output exactly
matching ustc_20class.npz's. Since A.dataset.stratified_splits only depends
on the label vector and its order, matching y is what proves the new
archives' train/val/test split is the same partition of the same flows the
byte-image tracks were scored on; matching m (independently computed from
the same packets, not derived from y) proves the re-walked PCAPs landed on
the same flow content, not just the same flow count.

Run directly, outside the top-level pytest.ini scope:

    PYTHONPATH=. .venv/bin/python -m pytest C/tests/
"""

import numpy as np
import pytest

REFERENCE = "data/processed/ustc_20class.npz"
CANDIDATES = [
    "data/processed/ustc_20class_flowpic_orig.npz",
    "data/processed/ustc_20class_fsnet_orig.npz",
]


def _load(path):
    with np.load(path, allow_pickle=True) as d:
        return d["y"], d["m"], list(d["labels"])


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_labels_and_stats_match_reference(candidate):
    ref_y, ref_m, ref_labels = _load(REFERENCE)
    cand_y, cand_m, cand_labels = _load(candidate)

    assert cand_labels == ref_labels
    assert np.array_equal(cand_y, ref_y)
    assert np.allclose(cand_m, ref_m)
