"""
Confirms C's test partition (A.dataset.stratified_splits) is identical to B's
(src.dataset.get_dataloader) for the same .npz at the shared split seed 62,
so all three sets of numbers are measured on the same held-out flows. Run
directly, outside the top-level pytest.ini scope:

    PYTHONPATH=. .venv/bin/python -m pytest C/tests/
"""

import numpy as np

from A.dataset import stratified_splits
from src.dataset import get_dataloader

NPZ_10CLASS = "data/processed/ustc_10class.npz"
SEED = 62


def _b_test_indices(npz_path, seed):
    # B's split only depends on the label vector, not on the tokenizer/prompt
    # settings, so a lightweight tokenizer keeps this fast.
    _, _, test_loader = get_dataloader(
        npz_path=npz_path,
        tokenizer="google-bert/bert-base-uncased",
        batch_size=64,
        seed=seed,
    )
    return sorted(test_loader.dataset.indices)


def _c_test_indices(npz_path):
    _, _, _, test_set = stratified_splits(npz_path, split_seed=SEED)
    return sorted(test_set.indices)


def test_c_test_split_matches_b():
    b_indices = _b_test_indices(NPZ_10CLASS, SEED)
    c_indices = _c_test_indices(NPZ_10CLASS)
    assert b_indices == c_indices


def test_c_split_is_deterministic_across_calls():
    first = _c_test_indices(NPZ_10CLASS)
    second = _c_test_indices(NPZ_10CLASS)
    assert first == second


def test_no_overlap_between_splits():
    dataset, train_set, val_set, test_set = stratified_splits(NPZ_10CLASS, split_seed=SEED)
    train_idx, val_idx, test_idx = set(train_set.indices), set(val_set.indices), set(test_set.indices)
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(dataset)
