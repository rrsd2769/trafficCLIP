"""
The 10-class archive is sliced out of the 20-class one rather than rebuilt,
to avoid re-parsing 3.7 GB of captures. This pins that the slice keeps the
right rows and renumbers labels correctly.
"""

import numpy as np
import pytest

from src.make_traffic_dataset import derive_subset


@pytest.fixture
def source(tmp_path):
    """A 4-class archive with 3 samples each, values encoding (class, sample)."""
    names = ["Alpha", "Bravo", "Charlie", "Delta"]
    x, y, m = [], [], []
    for k in range(4):
        for i in range(3):
            x.append(np.full((1, 28, 28), k * 10 + i, dtype=np.float32))
            y.append(k)
            m.append([k * 10 + i, 0.0, 0.0])
    path = tmp_path / "src.npz"
    np.savez_compressed(
        path,
        x=np.array(x),
        y=np.array(y),
        m=np.array(m),
        labels=names,
    )
    return path, names


def test_subset_keeps_only_requested_classes(source, tmp_path):
    path, _ = source
    out = derive_subset(path, tmp_path / "sub.npz", ["Charlie", "Alpha"])

    with np.load(out, allow_pickle=True) as d:
        x, y, m, labels = d["x"], d["y"], d["m"], list(d["labels"])

    assert list(labels) == ["Alpha", "Charlie"], "labels must be sorted"
    assert len(x) == 6
    assert sorted(set(y.tolist())) == [0, 1]


def test_labels_are_renumbered_over_the_subset(source, tmp_path):
    path, _ = source
    out = derive_subset(path, tmp_path / "sub.npz", ["Charlie", "Alpha"])

    with np.load(out, allow_pickle=True) as d:
        x, y, labels = d["x"], d["y"], list(d["labels"])

    # Alpha rows carried values 0,1,2; Charlie rows carried 20,21,22.
    for i in range(len(x)):
        encoded = float(x[i].flat[0])
        expected_source_class = int(encoded // 10)
        assert labels[y[i]] == ["Alpha", "Bravo", "Charlie", "Delta"][expected_source_class]


def test_rows_are_byte_identical_to_the_source(source, tmp_path):
    path, _ = source
    out = derive_subset(path, tmp_path / "sub.npz", ["Bravo"])

    with np.load(path, allow_pickle=True) as d:
        src_x, src_y, src_m = d["x"], d["y"], d["m"]
    with np.load(out, allow_pickle=True) as d:
        sub_x, sub_m = d["x"], d["m"]

    np.testing.assert_array_equal(sub_x, src_x[src_y == 1])
    np.testing.assert_array_equal(sub_m, src_m[src_y == 1])


def test_missing_class_is_rejected(source, tmp_path):
    path, _ = source
    with pytest.raises(ValueError, match="lacks classes"):
        derive_subset(path, tmp_path / "sub.npz", ["Alpha", "Echo"])
