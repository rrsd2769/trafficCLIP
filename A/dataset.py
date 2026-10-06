"""Dataset and stratified split for A."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

# Independent of the per-run training seed, so every run against a given .npz
# gets the same train/val/test partition and only model init and batch order
# vary between runs.
SPLIT_SEED = 62


class TrafficImageDataset(Dataset):
    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=True) as d:
            self.images = torch.from_numpy(d["x"]).float()
            self.labels = torch.from_numpy(d["y"]).long()
            self.class_names = list(d["labels"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx], "idx": idx}


def stratified_splits(npz_path, val_frac=0.15, test_frac=0.15, split_seed=SPLIT_SEED):
    """70/15/15 stratified train/val/test split, fixed at `split_seed`."""
    dataset = TrafficImageDataset(npz_path)
    targets = dataset.labels.tolist()

    train_idx, temp_idx = train_test_split(
        range(len(dataset)),
        test_size=val_frac + test_frac,
        stratify=targets,
        random_state=split_seed,
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp_targets,
        random_state=split_seed,
    )
    return dataset, Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
