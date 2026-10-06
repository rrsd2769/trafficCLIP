import logging
from pathlib import Path

import py7zr
import requests

from src.utils.utils import load_config

# -------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def download_file(url: str, save_path: Path) -> bool:
    """Download a file using streaming."""
    try:
        logging.info(f"Downloading to {save_path.name}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with save_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed: {e}")
        return False


def extract_7z(archive_path: Path, extract_path: Path) -> bool:
    """Extract a .7z archive."""
    try:
        logging.info(f"Extracting to {extract_path}")
        extract_path.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=extract_path)
        return True
    except py7zr.exceptions.ArchiveError as e:
        logging.error(f"Extraction failed: {e}")
        return False


def cleanup_file(file_path: Path) -> None:
    """Remove archive after successful extraction."""
    try:
        file_path.unlink()
        logging.info(f"Cleanup complete for {file_path.name}")
    except Exception as e:
        logging.warning(f"Cleanup failed for {file_path}: {e}")


# -------------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------------
def download_and_extract(base_url: str, data_dir: Path, classes: dict) -> None:
    """
    Downloads and extracts datasets based on YAML dictionary structure.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    for category, class_list in classes.items():
        for item in class_list:
            # Extract name and extension from the YAML dictionary
            class_name = item["name"]
            extension = item["extension"]

            logging.info(f"--- Processing {class_name} ({category}) ---")

            file_name = f"{class_name}{extension}"
            url = f"{base_url}/{category}/{file_name}"

            target_path = data_dir / file_name
            extract_path = data_dir / class_name

            # 1. Download the file
            if not download_file(url, target_path):
                logging.error(f"Skipping {class_name} due to download failure.")
                continue

            # 2. Handle Extraction/Organization
            try:
                if extension == ".7z":
                    if extract_7z(target_path, extract_path):
                        cleanup_file(target_path)
                    else:
                        continue
                else:
                    # For .pcap and others, move into dedicated folder
                    extract_path.mkdir(parents=True, exist_ok=True)
                    destination = extract_path / file_name
                    target_path.replace(destination)
                    logging.info(f"Moved {file_name} to {extract_path}")

                logging.info(f"Successfully prepared: {class_name}\n")

            except Exception as e:
                logging.error(f"Error processing {class_name}: {str(e)}")
                continue


# -------------------------------------------------------------------------
# Script Entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        config = load_config()
        BASE_URL = config["dataset"]["traffic"]["url"]
        DATA_DIR = Path(config["paths"]["raw_data_dir"])
        CLASSES = config["dataset"]["traffic"]["classes"]

        download_and_extract(BASE_URL, DATA_DIR, CLASSES)
    except Exception as e:
        logging.critical(f"Pipeline failed: {str(e)}")
