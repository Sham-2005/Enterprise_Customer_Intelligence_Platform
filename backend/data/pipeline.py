"""
ETL & Feature Pipeline Orchestrator for ECIP.
Executes end-to-end data ingestion, validation, cleaning, merging, feature engineering,
label generation, data quality reporting, and data warehouse persistence.
"""

import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from config.settings import Settings
from backend.data.loader import DataLoader
from backend.data.validator import DataValidator
from backend.data.preprocessor import DataPreprocessor
from backend.data.merger import DataMerger
from backend.data.feature_engineer import FeatureEngineer
from backend.data.label_engineer import LabelEngineer
from backend.data.quality_reporter import DataQualityReporter
from utils.logger import setup_logger

logger = setup_logger("ECIP.ETLPipeline")

class ETLPipeline:
    """End-to-end Data Engineering and Feature Store Pipeline Orchestrator."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.output_dir = self.settings.get_path("paths.output_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader(config_path)
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
        self.merger = DataMerger()
        self.feature_engineer = FeatureEngineer()
        self.label_engineer = LabelEngineer(config_path)
        self.quality_reporter = DataQualityReporter(config_path)

    def run(self) -> Dict[str, Path]:
        """
        Executes complete ETL & Feature Engineering pipeline.

        Returns:
            Dict mapping output dataset names to saved CSV file paths.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STARTING ENTERPRISE DATA & FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Ingestion
        t0 = time.time()
        datasets = self.loader.load_all_datasets()
        logger.info(f"[Stage 1/7] Ingestion finished in {time.time() - t0:.2f}s")

        # Step 2: Validation
        t0 = time.time()
        validation_report = self.validator.validate_datasets(datasets)
        logger.info(f"[Stage 2/7] Data Validation finished in {time.time() - t0:.2f}s")

        # Step 3: Cleaning & Preprocessing
        t0 = time.time()
        cleaned_datasets = self.preprocessor.clean_datasets(datasets)
        logger.info(f"[Stage 3/7] Preprocessing & Cleaning finished in {time.time() - t0:.2f}s")

        # Step 4: Relational Merging
        t0 = time.time()
        master_df = self.merger.merge_datasets(cleaned_datasets)
        logger.info(f"[Stage 4/7] Master Dataset Merging finished in {time.time() - t0:.2f}s (Shape: {master_df.shape})")

        # Step 5: Feature Engineering
        t0 = time.time()
        cust_features, prod_features, order_features, seller_features = (
            self.feature_engineer.compute_all_features(master_df)
        )
        logger.info(f"[Stage 5/7] Feature Engineering finished in {time.time() - t0:.2f}s")

        # Step 6: Derived ML Labels & Segmentation
        t0 = time.time()
        feature_store_df = self.label_engineer.generate_all_labels(cust_features)
        logger.info(f"[Stage 6/7] Label Engineering & RFM Segmentation finished in {time.time() - t0:.2f}s")

        # Step 7: Persistence & Data Warehouse Export
        t0 = time.time()
        saved_paths = self._export_data_warehouse(master_df, feature_store_df)

        # Generate Data Quality Reports (HTML and CSV)
        self.quality_reporter.generate_report(validation_report, feature_store_df)
        logger.info(f"[Stage 7/7] Persistence & Quality Reporting finished in {time.time() - t0:.2f}s")

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"ETL PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
        logger.info(f"Total Unique Customers Processed: {len(feature_store_df):,}")
        logger.info(f"Total Master Relational Rows: {len(master_df):,}")
        logger.info("=" * 60)

        return saved_paths

    def _export_data_warehouse(
        self, master_df: pd.DataFrame, feature_store_df: pd.DataFrame
    ) -> Dict[str, Path]:
        """Saves versioned Data Warehouse CSV datasets to output directory."""
        paths = {}

        # 1. master_dataset.csv
        master_path = self.output_dir / "master_dataset.csv"
        master_df.to_csv(master_path, index=False)
        paths["master_dataset"] = master_path
        logger.info(f"Saved Master Dataset ({len(master_df)} rows) to {master_path}")

        # 2. feature_store.csv
        feature_path = self.output_dir / "feature_store.csv"
        feature_store_df.to_csv(feature_path, index=False)
        paths["feature_store"] = feature_path
        logger.info(f"Saved Feature Store ({len(feature_store_df)} rows, {len(feature_store_df.columns)} features) to {feature_path}")

        # 3. processed_customers.csv
        cust_cols = [
            "customer_unique_id", "total_orders", "total_spending",
            "avg_order_value", "recency_days", "churn_label", "spending_tier"
        ]
        cust_path = self.output_dir / "processed_customers.csv"
        feature_store_df[cust_cols].to_csv(cust_path, index=False)
        paths["processed_customers"] = cust_path

        # 4. rfm_dataset.csv
        rfm_cols = [
            "customer_unique_id", "recency_days", "total_orders", "total_spending",
            "recency_score", "frequency_score", "monetary_score", "rfm_combined", "rfm_segment"
        ]
        rfm_path = self.output_dir / "rfm_dataset.csv"
        feature_store_df[rfm_cols].to_csv(rfm_path, index=False)
        paths["rfm_dataset"] = rfm_path

        # 5. customer_metrics.csv
        metrics_cols = [
            "customer_unique_id", "historical_clv", "clv_per_order", "loyalty_score",
            "avg_review_score_given", "is_repeat_customer", "preferred_payment_method"
        ]
        metrics_path = self.output_dir / "customer_metrics.csv"
        feature_store_df[metrics_cols].to_csv(metrics_path, index=False)
        paths["customer_metrics"] = metrics_path

        return paths
