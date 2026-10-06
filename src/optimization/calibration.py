import logging

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.dataset import get_dataloader
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def create_calibration_dataloader(num_samples=200):
    # load config
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    try:
        NPZ_PATH = config["paths"]["output_data_file"]
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = config["train"]["seed"]
        BATCH_SIZE = config["preprocess"]["batch_size"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    _, val_loader_optimized, _ = get_dataloader(
        npz_path=NPZ_PATH,
        tokenizer=TOKENIZER_NAME,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        seed=SEED,
        use_dynamic_prompts=True,
    )

    val_dataset = val_loader_optimized.dataset

    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    calibration_subset = Subset(val_dataset, indices)
    calibration_dataloader = DataLoader(
        calibration_subset, batch_size=32, shuffle=False
    )

    logging.info(
        f"Calibration loader created with {len(calibration_subset)} samples from the Validation set."
    )
    return calibration_dataloader


def create_profiling_dataloader(num_samples=500):
    """
    Defect 2 fix: profiling.py benchmarked against `create_calibration_dataloader`,
    which subsamples the *validation* split, even though the manuscript's Table 2
    says "500 test samples". This draws from the held-out test split instead.
    """
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    try:
        NPZ_PATH = config["paths"]["output_data_file"]
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = config["train"]["seed"]
        BATCH_SIZE = config["preprocess"]["batch_size"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    _, _, test_loader = get_dataloader(
        npz_path=NPZ_PATH,
        tokenizer=TOKENIZER_NAME,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        seed=SEED,
        use_dynamic_prompts=True,
    )

    test_dataset = test_loader.dataset

    indices = np.random.choice(
        len(test_dataset), min(num_samples, len(test_dataset)), replace=False
    )
    profiling_subset = Subset(test_dataset, indices)
    profiling_dataloader = DataLoader(profiling_subset, batch_size=32, shuffle=False)

    logging.info(
        f"Profiling loader created with {len(profiling_subset)} samples from the Test set."
    )
    return profiling_dataloader
