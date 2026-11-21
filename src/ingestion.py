"""
Module: ingestion.py
Description: Ingests the UCI Student Performance Dataset, maintains raw layer,
             and records metadata (extraction date, source, status, row count).
"""

import os
import zipfile
import urllib.request
import json
import logging
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
STAGING_DIR = os.path.join(DATA_DIR, 'staging')

# Dataset constants
UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
ZIP_FILE_PATH = os.path.join(RAW_DIR, "student.zip")
RAW_MATH_CSV = os.path.join(RAW_DIR, "student-mat.csv")
RAW_POR_CSV = os.path.join(RAW_DIR, "student-por.csv")
INGESTION_LOG_PATH = os.path.join(RAW_DIR, "ingestion_log.json")
STAGING_CSV_PATH = os.path.join(STAGING_DIR, "student_staging.csv")


def ensure_directories():
    """Ensure all required data layer directories exist."""
    for d in [
        RAW_DIR,
        STAGING_DIR,
        os.path.join(DATA_DIR, 'cleaned'),
        os.path.join(DATA_DIR, 'rejected'),
        os.path.join(DATA_DIR, 'analytics'),
        os.path.join(BASE_DIR, 'docs'),
        os.path.join(BASE_DIR, 'sql'),
        os.path.join(BASE_DIR, 'dashboard'),
        os.path.join(BASE_DIR, 'tests'),
    ]:
        os.makedirs(d, exist_ok=True)


def download_uci_dataset(url: str = UCI_ZIP_URL, target_zip: str = ZIP_FILE_PATH) -> bool:
    """Download the raw zip archive from UCI repository if not already present."""
    if os.path.exists(RAW_MATH_CSV):
        logger.info(f"Raw dataset already exists at {RAW_MATH_CSV}. Skipping download.")
        return True

    logger.info(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, target_zip)
        logger.info(f"Downloaded ZIP archive to {target_zip}")
        
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        logger.info(f"Extracted dataset files into {RAW_DIR}")
        
        if os.path.exists(target_zip):
            os.remove(target_zip)
        return True
    except Exception as e:
        logger.error(f"Failed to download/extract dataset: {e}")
        return False


def ingest_data(source_url: str = UCI_ZIP_URL) -> dict:
    """
    Main ingestion execution:
    1. Ensures directories exist.
    2. Downloads and stores raw dataset (unmodified).
    3. Reads and verifies row count.
    4. Stages initial data with unique IDs.
    5. Writes extraction metadata log.
    """
    ensure_directories()
    start_time = datetime.now()
    status = "SUCCESS"
    error_msg = ""
    row_count = 0
    col_count = 0

    try:
        # Download if needed
        success = download_uci_dataset(source_url)
        if not success and not os.path.exists(RAW_MATH_CSV):
            raise FileNotFoundError(f"Raw CSV not found at {RAW_MATH_CSV} and download failed.")

        # Read raw dataset without modification (delimiter in UCI dataset is ';')
        logger.info(f"Reading raw CSV from {RAW_MATH_CSV}...")
        df_raw = pd.read_csv(RAW_MATH_CSV, sep=';')
        row_count, col_count = df_raw.shape
        logger.info(f"Raw data successfully ingested. Rows: {row_count}, Columns: {col_count}")


        df_staging = df_raw.copy()
        if 'student_id' not in df_staging.columns:
            df_staging.insert(0, 'student_id', [f"STU{str(i).zfill(4)}" for i in range(1, len(df_staging) + 1)])
        
        df_staging.to_csv(STAGING_CSV_PATH, index=False)
        logger.info(f"Staged copy written to {STAGING_CSV_PATH}")

    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        logger.error(f"Ingestion process failed: {e}")

    # Metadata logging
    metadata = {
        "dataset_name": "UCI Student Performance Dataset (Mathematics)",
        "source": source_url,
        "extraction_date": start_time.isoformat(),
        "status": status,
        "raw_file": os.path.relpath(RAW_MATH_CSV, BASE_DIR),
        "row_count": row_count,
        "column_count": col_count,
        "error_message": error_msg
    }

    with open(INGESTION_LOG_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Ingestion metadata recorded in {INGESTION_LOG_PATH}")

    return metadata


if __name__ == "__main__":
    result = ingest_data()
    print("Ingestion Result:", json.dumps(result, indent=2))
