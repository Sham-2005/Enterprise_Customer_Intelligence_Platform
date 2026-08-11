"""
Enterprise Data Service for ECIP Executive Dashboard.
Automatically detects dataset availability in output directory, parses date schemas,
handles missing datasets gracefully, and provides clean dataframes for analytics.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from config.settings import Settings
from backend.cache.dashboard_cache import dashboard_cache
from utils.logger import setup_logger

logger = setup_logger("ECIP.DataService")

class DataService:
    """Service for discovering, loading, caching, and managing enterprise datasets."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.output_dir = self.settings.get_path("paths.output_dir")
        self.dataset_dir = self.settings.get_path("paths.dataset_dir")

    def get_dataset_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans output directory to verify availability and file metadata for key processed datasets.
        """
        expected_files = {
            "master_dataset": "master_dataset.csv",
            "feature_store": "feature_store.csv",
            "customer_metrics": "customer_metrics.csv",
            "churn_predictions": ["customer_churn_predictions.csv", "churn_predictions.csv"],
            "clv_predictions": ["customer_clv_predictions.csv", "clv_predictions.csv"],
            "processed_customers": "processed_customers.csv",
            "rfm_dataset": "rfm_dataset.csv"
        }

        status = {}
        for key, fname in expected_files.items():
            found_path = None
            if isinstance(fname, list):
                for candidate in fname:
                    p = self.output_dir / candidate
                    if p.exists():
                        found_path = p
                        break
            else:
                p = self.output_dir / fname
                if p.exists():
                    found_path = p

            if found_path and found_path.exists():
                stat = found_path.stat()
                status[key] = {
                    "available": True,
                    "path": str(found_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "last_modified": pd.Timestamp(stat.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                status[key] = {
                    "available": False,
                    "path": str(self.output_dir / (fname[0] if isinstance(fname, list) else fname)),
                    "size_mb": 0.0,
                    "last_modified": "N/A"
                }

        return status

    def load_all_executive_datasets(self, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Loads all available processed datasets into memory with caching.
        Handles missing files gracefully by returning empty/synthetic structure or raw merged fallback.
        """
        if not force_reload:
            try:
                from dashboard.utils.cache_manager import get_cached_all_datasets
                return get_cached_all_datasets()
            except Exception as e:
                logger.warning(f"Streamlit cache manager unavailable, using internal cache: {e}")

        cache_key_prefix = "executive_datasets_all"
        if not force_reload:
            cached = dashboard_cache.get(cache_key_prefix)
            if cached is not None:
                return cached

        logger.info("Loading executive datasets from output storage...")
        datasets: Dict[str, pd.DataFrame] = {}
        status = self.get_dataset_status()

        for key, meta in status.items():
            if meta["available"]:
                try:
                    path = Path(meta["path"])
                    df = pd.read_csv(path)
                    df = self._parse_date_columns(df)
                    datasets[key] = df
                    logger.info(f"Loaded dataset '{key}' successfully with shape {df.shape}")
                except Exception as e:
                    logger.error(f"Failed to load dataset file '{meta['path']}': {e}")
                    datasets[key] = pd.DataFrame()
            else:
                logger.warning(f"Dataset '{key}' is missing at {meta['path']}.")
                datasets[key] = pd.DataFrame()

        # If master_dataset is missing, attempt on-the-fly execution or raw dataset loader fallback
        if datasets.get("master_dataset", pd.DataFrame()).empty:
            logger.warning("master_dataset.csv was missing. Attempting raw dataset fallback ingestion...")
            datasets["master_dataset"] = self._attempt_raw_fallback_ingestion()

        dashboard_cache.set(cache_key_prefix, datasets, ttl=3600)
        return datasets

    def _parse_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to convert date and timestamp string columns to datetime objects."""
        if df.empty:
            return df

        date_keywords = ["timestamp", "date", "year", "month", "day"]
        for col in df.columns:
            if any(kw in col.lower() for kw in date_keywords):
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    except Exception:
                        pass
        return df

    def _attempt_raw_fallback_ingestion(self) -> pd.DataFrame:
        """
        Constructs a lightweight merged dataset directly from raw data_set/ files
        if output/master_dataset.csv has not yet been built by run_pipeline.py.
        """
        try:
            orders_p = self.dataset_dir / self.settings.get("dataset_files.orders")
            items_p = self.dataset_dir / self.settings.get("dataset_files.order_items")
            cust_p = self.dataset_dir / self.settings.get("dataset_files.customers")
            payments_p = self.dataset_dir / self.settings.get("dataset_files.payments")
            reviews_p = self.dataset_dir / self.settings.get("dataset_files.reviews")
            products_p = self.dataset_dir / self.settings.get("dataset_files.products")
            trans_p = self.dataset_dir / self.settings.get("dataset_files.category_translation")

            if orders_p.exists() and items_p.exists() and cust_p.exists():
                orders = pd.read_csv(orders_p)
                items = pd.read_csv(items_p)
                cust = pd.read_csv(cust_p)
                
                merged = items.merge(orders, on="order_id", how="inner").merge(cust, on="customer_id", how="inner")

                if payments_p.exists():
                    pmt = pd.read_csv(payments_p).groupby("order_id").first().reset_index()
                    merged = merged.merge(pmt[["order_id", "payment_type", "payment_value"]], on="order_id", how="left")

                if reviews_p.exists():
                    rev = pd.read_csv(reviews_p).groupby("order_id")["review_score"].mean().reset_index()
                    rev.rename(columns={"review_score": "avg_review_score"}, inplace=True)
                    merged = merged.merge(rev, on="order_id", how="left")

                if products_p.exists() and trans_p.exists():
                    prod = pd.read_csv(products_p)
                    trans = pd.read_csv(trans_p)
                    prod = prod.merge(trans, on="product_category_name", how="left")
                    merged = merged.merge(prod[["product_id", "product_category_name_english"]], on="product_id", how="left")

                merged = self._parse_date_columns(merged)
                logger.info(f"Fallback raw data ingestion constructed master dataset shape: {merged.shape}")
                return merged
        except Exception as e:
            logger.error(f"Fallback raw ingestion failed: {e}")

        return pd.DataFrame()
