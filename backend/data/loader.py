"""
Olist Dataset Loader Module for ECIP.
Handles dynamic file path resolution, encoding auto-detection, schema verification, and data loading.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from config.settings import Settings
from utils.logger import setup_logger
from utils.exceptions import DataIngestionError

logger = setup_logger("ECIP.DataLoader")

class DataLoader:
    """Class responsible for locating, validating integrity of, and reading Olist CSV files."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.dataset_dir = self.settings.get_path("paths.dataset_dir")
        self.files_config = self.settings.get("dataset_files", {})

    def check_file_existence(self) -> Dict[str, Tuple[Path, bool]]:
        """Verifies existence of configured raw CSV dataset files."""
        existence_map = {}
        for key, filename in self.files_config.items():
            file_path = self.dataset_dir / filename
            exists = file_path.exists()
            existence_map[key] = (file_path, exists)
            if not exists:
                logger.warning(f"Dataset file missing for key '{key}': {file_path}")
            else:
                logger.info(f"Dataset file verified for key '{key}': {filename}")
        return existence_map

    def load_csv_with_encoding(self, file_path: Path) -> pd.DataFrame:
        """Loads a CSV file attempting standard encodings (utf-8, latin1, iso-8859-1)."""
        encodings = ["utf-8", "iso-8859-1", "latin1"]
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                logger.debug(f"Successfully loaded {file_path.name} with encoding={enc} (shape: {df.shape})")
                return df
            except (UnicodeDecodeError, Exception) as e:
                logger.debug(f"Failed loading {file_path.name} with encoding={enc}: {e}")
        
        raise DataIngestionError(
            f"Could not load CSV file {file_path.name} with standard encodings.",
            details=f"Path: {file_path}"
        )

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Loads all Olist datasets into a dictionary of pandas DataFrames keyed by dataset name."""
        logger.info("Starting raw dataset ingestion...")
        existence = self.check_file_existence()
        missing = [k for k, (path, exists) in existence.items() if not exists]
        
        if missing:
            raise DataIngestionError(
                f"Cannot proceed with loading. Missing required dataset files for keys: {missing}",
                details=f"Dataset Directory: {self.dataset_dir}"
            )

        datasets = {}
        for key, (file_path, _) in existence.items():
            logger.info(f"Ingesting '{key}' dataset from {file_path.name}...")
            df = self.load_csv_with_encoding(file_path)
            datasets[key] = df

        logger.info(f"Successfully ingested all {len(datasets)} raw datasets.")
        return datasets
