import logging
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_device(prefer: str | None = None) -> torch.device:
    """
    Selects the best available compute device.

    Order of preference: CUDA, then Apple MPS, then CPU. Every entry point in
    this project previously hardcoded `cuda`-or-`cpu`, which silently resolved
    to CPU on Apple silicon and left the GPU unused.

    Args:
        prefer: force a specific device string (e.g. "cpu"). Used by the ONNX
            export and quantization paths, which must stay on CPU because the
            MPS backend does not implement the quantized operators.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path="config/config.yaml"):
    """
    Load YAML config file from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def unzip_and_extract(zip_file_path_str: str, extraction_dir_str: str) -> bool:
    """
    Unzips a file to a specified directory.
    """
    zip_path = Path(zip_file_path_str)
    extraction_dir = Path(extraction_dir_str)

    # Check if the source zip file exists
    if not zip_path.is_file():
        logging.error(f"Source file not found: {zip_path}")
        return False

    # Ensure the extraction directory exists
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # Extract the contents
    try:
        logging.info(f"Extracting '{zip_path.name}' to '{extraction_dir}'...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extraction_dir)
        logging.info("Extraction successful.")
        return True
    except zipfile.BadZipFile:
        logging.error(f"Error: '{zip_path}' is not a valid zip file.")
        return False
    except Exception:
        logging.exception(f"An unexpected error occurred during extraction.")
        return False


def export_data(dataframe: pd.DataFrame, export_path: str, name: str) -> None:
    """
    Exports a DataFrame to a CSV file.
    """
    file_path = Path(export_path) / f"{name}.csv"

    try:
        dataframe.to_csv(file_path, index=False)
        print(f"Data exported to: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to export data to {file_path}") from e


def get_metrics(true, pred):
    """Calculates common regression metrics between true and predicted values.
    Parameters:
        true (np.ndarray): Array of true target values.
        pred (np.ndarray): Array of predicted values.
    Returns:
        tuple: A tuple containing the following metrics:
            - mae (float): Mean Absolute Error.
            - mse (float): Mean Squared Error.
            - rmse (float): Root Mean Squared Error.
            - mape (float): Mean Absolute Percentage Error (computed only where true > 0).
    Notes:
        - MAPE is calculated only for elements where the true value is greater than zero to avoid division by zero.
        - All metrics are computed using numpy operations.
    """

    # Mean Absolute Error
    mae = np.mean(np.abs(true - pred))

    # Mean Squared Error
    mse = np.mean((true - pred) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (Handling Zeros)
    # np.where to avoid division by zero
    # Calculates MAPE only for cases where true demand > 0
    mask = true > 0
    if np.any(mask):
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    else:
        mape = 0.0

    return mae, mse, rmse, mape


def plot_convergence(history, model_type, save_path: Path = None):
    """
    Plots training and validation loss and F1 score over epochs.
    Parameters:
        history (dict): Dictionary containing 'train_loss', 'val_loss', and 'val_f1' lists.
        model_type (str): Type of the model for labeling the plots.
        save_path (Path, optional): Path to save the figure. If None, saves to default location.
    Returns:
        fig: Matplotlib figure object.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title(f"{model_type} Loss Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss (CE + CL)")
    plt.legend()
    plt.grid(True)

    # F1 Score Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_f1"], "g-", label="Val Macro F1")
    plt.title(f"{model_type} F1 Score Progress")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # save figure
    if save_path is None:
        save_path = Path(__file__).parent.parent.parent / "results" / "figures"
    plt.savefig(save_path / f"{model_type}_convergence.png")
    plt.close()
    return fig


# def plot_confusion_matrix(y_true, y_pred, class_names, model_type, model_version):
#     target_labels = np.arange(len(class_names))
#     cm = confusion_matrix(y_true, y_pred, labels=target_labels)
#     fig = plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt=".1f",  # Changed from 'd' to '.1f' to handle float values
#         cmap="Blues",
#         xticklabels=class_names,
#         yticklabels=class_names,
#     )
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.title(f"Confusion Matrix: {model_type}")
#     # save figure
#     figure_path = (
#         Path(__file__).parent.parent.parent / "results" / "figures" / model_version
#     )
#     figure_path.mkdir(parents=True, exist_ok=True)
#     plt.savefig(figure_path / f"{model_type}_confusion_matrix.png")
#     plt.close()
#     return fig
def plot_confusion_matrix(y_true, y_pred, class_names, model_type, model_version):
    # Ensure inputs are numpy arrays for indexing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    target_labels = np.arange(len(class_names))

    # 1. Calculate raw and normalized matrices
    cm = confusion_matrix(y_true, y_pred, labels=target_labels)
    # Normalize by row (true labels) to get percentages
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero for empty classes

    fig, ax = plt.subplots(figsize=(12, 10))

    # 2. Use the normalized matrix for colors, but show raw counts as labels
    sns.heatmap(
        cm_norm,
        annot=cm,  # <--- This puts the raw counts (1, 2, etc.) inside the boxes
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.title(
        f"Confusion Matrix: {model_type} ({model_version})\nSamples: {len(y_true)}",
        fontsize=14,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # 3. Handle paths
    figure_path = (
        Path(__file__).parent.parent.parent / "results" / "figures" / model_version
    )
    figure_path.mkdir(parents=True, exist_ok=True)

    save_file = figure_path / f"{model_type}_confusion_matrix.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    logging.info(f"Confusion Matrix saved to: {save_file}")

    return fig


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def save_metrics(run_metrics, model_type, model_version):
    """
    Saves multi-seed metrics to a CSV file in an isolated directory.

    Args:
        run_metrics (list): List of [Acc, Prec, Rec, F1] from each seed run.
        model_type (str): Unique tag for the specific run (e.g., 'optimized_L2.0_statsTrue').
        model_version (str): Higher-level model category ('original' or 'optimized').
    """
    # Construct the path: results/metrics/optimized/
    results_path = (
        Path(__file__).parent.parent.parent / "results" / "metrics" / model_version
    )
    results_path.mkdir(parents=True, exist_ok=True)

    # 2. Define the filename based on the unique experiment tag
    results_file = results_path / f"{model_type}_results.csv"

    # 3. Convert metrics to a DataFrame and save
    df = pd.DataFrame(
        run_metrics, columns=["Accuracy", "Precision", "Recall", "Macro F1"]
    )
    df.to_csv(results_file, index_label="Run")

    logging.info(f"Saved detailed run metrics to {results_file}")
    return results_file


def get_original_descriptor_bank(model, tokenizer, class_names, max_length, device):
    """
    Creates a static bank of class embeddings for the Original Prompt variant.
    """
    prompts = [f"A network traffic gray photo of class {name}" for name in class_names]
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        # Shape: [K_classes, 1024]
        text_features = model.get_text_features(
            encoded["input_ids"], encoded["attention_mask"]
        )
    return text_features


if __name__ == "__main__":
    load_config()
