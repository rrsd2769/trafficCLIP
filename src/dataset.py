import logging
import pathlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class TrafficDataset(Dataset):
    def __init__(
        self,
        npz_path,
        # prompts,
        tokenizer_name="google/bert-base-uncased",
        max_length=64,
        use_dynamic_prompts=False,
    ):
        data = np.load(npz_path, allow_pickle=True)

        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.use_dynamic_prompts = use_dynamic_prompts

        # Image Tensors: (N, 1, 28, 28)
        self.images = torch.from_numpy(data["x"]).float()
        self.labels = torch.from_numpy(data["y"]).long()
        self.class_names = data["labels"].tolist()

        # # Pre-calculate prompts
        # self.raw_prompts = [prompts[name] for name in self.class_names]

        self.metadata = data.get("m", None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        class_name = self.class_names[label]
        # text_description = self.raw_prompts[label]
        m_iat, m_jitter, m_entropy = self.metadata[idx]

        if self.use_dynamic_prompts and self.metadata is not None:
            # Physics-Informed Dynamic Prompt
            text_description = (
                f"A network traffic gray photo of class {class_name} with "
                f"{m_iat:.2f}ms mean IAT, {m_jitter:.2f}ms jitter, "
                f"and {m_entropy:.2f} byte entropy."
            )
        else:
            text_description = f"A network traffic gray photo of class {class_name}."

        # Tokenize prompt
        tokens = self.tokenizer(
            text_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create the 3-dim stats_vector
        stats_vector = torch.tensor([m_iat, m_jitter, m_entropy], dtype=torch.float32)

        return {
            "image": image,  # Input for Vision Encoder
            "input_ids": tokens["input_ids"].squeeze(0),  # Input for BERT Encoder
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "stats": stats_vector,  # Added stats key
            "label": label,
            "raw_text": text_description,  # debugging
            "idx": idx,  # global row index, for addressing the feature cache
        }


def get_dataloader(
    npz_path,
    tokenizer,
    batch_size=64,
    max_length=64,
    seed=42,
    use_dynamic_prompts=False,
):
    """
    Creates Stratified Train, Validation, and Test loaders.
    Ensures class distribution is preserved across all splits.
    """
    full_dataset = TrafficDataset(
        npz_path,
        tokenizer_name=tokenizer,
        max_length=max_length,
        use_dynamic_prompts=use_dynamic_prompts,
    )

    # Extract all labels to perform stratification.
    # Read straight off the label tensor rather than calling __getitem__, which
    # would run the BERT tokenizer once per sample -- 18,000 tokenizer calls to
    # obtain integers that are already sitting in memory.
    targets = full_dataset.labels.tolist()

    # 70% Train, 30% for Val + Test
    train_indices, temp_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.30,
        stratify=targets,
        random_state=seed,
    )

    # Split the 30% into 15% Val and 15% Test
    temp_targets = [targets[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,  # Half of 30% is 15%
        stratify=temp_targets,
        random_state=seed,
    )

    # Create PyTorch Subsets using the stratified indices
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)

    # Create DataLoaders
    # Fix: shuffle=False froze batch composition across every epoch, which
    # specifically degrades the contrastive loss (each anchor sees the same
    # fixed set of positives/negatives every epoch instead of a resampled
    # one). Only the train loader needs shuffling; val/test order doesn't
    # affect the aggregate metrics computed over them.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Logging distribution
    logging.info(f"Total Samples: {len(full_dataset)}")
    logging.info(f"Stratified Training:   {len(train_set)} samples")
    logging.info(f"Stratified Validation: {len(val_set)} samples")
    logging.info(f"Stratified Testing:    {len(test_set)} samples")

    return train_loader, val_loader, test_loader


# Utility function to check class distribution in a DataLoader
def check_distribution(loader, name):
    all_labels = []
    for batch in loader:
        all_labels.extend(batch["label"].tolist())
    counts = Counter(all_labels)
    logging.info(f"{name} distribution: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the stratified splits.")
    parser.add_argument(
        "--npz",
        default=None,
        help="Path to the .npz archive. Defaults to paths.output_data_file.",
    )
    args = parser.parse_args()

    config = load_config()
    try:
        # Parameter Extraction
        # SEMANTIC_PROMPTS = config["prompts"]
        # class_names = config["prompts"].keys()
        # template = "A network traffic gray photo of {}"
        # ORIGINAL_PROMPTS = {label: template.format(label) for label in class_names}

        NPZ_PATH = args.npz or config["paths"]["output_data_file"]
        TENSOR_DIR = Path(config["paths"]["tensors_dir"])
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = 62
        BATCH_SIZE = config["preprocess"]["batch_size"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    try:
        train_loader, val_loader, test_loader = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
            use_dynamic_prompts=True,
            # prompts=SEMANTIC_PROMPTS,
        )

        logging.info("DataLoaders created successfully.")
        check_distribution(train_loader, "Train")
        check_distribution(val_loader, "Validation")
        check_distribution(test_loader, "Test")

    except Exception as e:
        logging.error(f"Error creating DataLoaders: {e}")
        raise
