"""
Dataset for the tri-modal variant.

Adds the per-sample stats array (`m` in the .npz: IAT, jitter, entropy) that
the base dataset does not load. Split logic matches A/dataset.py, so a
tri-modal run and a base run on the same .npz share a partition.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from A.dataset import SPLIT_SEED

DYNAMIC_PROMPT_TEMPLATE = (
    "A network traffic gray photo of class {name} with {iat:.2f}ms mean IAT, "
    "{jitter:.2f}ms jitter, and {entropy:.2f} byte entropy."
)


class TrafficStatsDataset(Dataset):
    """Images + labels + per-sample stats vector (IAT, jitter, entropy)."""

    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=True) as d:
            self.images = torch.from_numpy(d["x"]).float()
            self.labels = torch.from_numpy(d["y"]).long()
            self.class_names = list(d["labels"])
            self.stats = torch.from_numpy(d["m"]).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "stats": self.stats[idx],
            "idx": idx,
        }


def stratified_splits_full(npz_path, val_frac=0.15, test_frac=0.15, split_seed=SPLIT_SEED):
    """70/15/15 stratified split, identical partition logic to A/dataset.py."""
    dataset = TrafficStatsDataset(npz_path)
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


def build_dynamic_prompts(class_names, stats_batch):
    """
    Flattened B*K list of prompts for one batch: every image against every
    candidate class name, using that image's own stats. Only the class name
    varies across a sample's K hypotheses. Order is sample-major, so the
    embeddings reshape back with a plain .view(b, k, -1).
    """
    stats_np = stats_batch.detach().cpu().numpy()
    return [
        DYNAMIC_PROMPT_TEMPLATE.format(
            name=name.replace("_", " "), iat=stats_np[i, 0], jitter=stats_np[i, 1], entropy=stats_np[i, 2]
        )
        for i in range(stats_np.shape[0])
        for name in class_names
    ]
