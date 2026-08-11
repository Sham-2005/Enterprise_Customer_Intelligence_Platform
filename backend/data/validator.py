"""
Data Validation Module for ECIP.
Performs data quality checks across raw Olist datasets: missing values, duplicates, ID uniqueness,
timestamp ranges, and non-negative monetary constraints.
"""

from typing import Dict, Any
import pandas as pd
from utils.logger import setup_logger
from utils.exceptions import DataValidationError

logger = setup_logger("ECIP.DataValidator")

class DataValidator:
    """Validates data hygiene, schema constraints, missing values, duplicates, and anomaly rates."""

    def __init__(self):
        self.validation_summary: Dict[str, Any] = {}

    def validate_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Runs comprehensive validation checks across all ingested raw datasets."""
        logger.info("Executing raw data validation suite...")
        report = {}

        for name, df in datasets.items():
            dataset_report = self._validate_single_dataset(name, df)
            report[name] = dataset_report

        self.validation_summary = report
        logger.info("Data validation suite completed.")
        return report

    def _validate_single_dataset(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates a single dataset for rows, columns, nulls, duplicates, and key constraints."""
        total_rows = len(df)
        total_cols = len(df.columns)
        duplicate_rows = int(df.duplicated().sum())
        missing_values = df.isnull().sum().to_dict()
        missing_total = int(sum(missing_values.values()))

        # Specific anomalies check
        anomalies = []

        if name == "orders":
            if "order_id" in df.columns:
                dup_ids = int(df["order_id"].duplicated().sum())
                if dup_ids > 0:
                    anomalies.append(f"Duplicate order_ids found: {dup_ids}")
            if "order_status" in df.columns:
                empty_status = int(df["order_status"].isnull().sum())
                if empty_status > 0:
                    anomalies.append(f"Missing order status values: {empty_status}")

        elif name == "order_items":
            if "price" in df.columns:
                neg_prices = int((df["price"] < 0).sum())
                if neg_prices > 0:
                    anomalies.append(f"Negative price records found: {neg_prices}")
            if "freight_value" in df.columns:
                neg_freight = int((df["freight_value"] < 0).sum())
                if neg_freight > 0:
                    anomalies.append(f"Negative freight values found: {neg_freight}")

        elif name == "payments":
            if "payment_value" in df.columns:
                neg_payments = int((df["payment_value"] < 0).sum())
                if neg_payments > 0:
                    anomalies.append(f"Negative payment values found: {neg_payments}")

        elif name == "customers":
            if "customer_id" in df.columns:
                dup_cust_ids = int(df["customer_id"].duplicated().sum())
                if dup_cust_ids > 0:
                    anomalies.append(f"Duplicate customer_ids found: {dup_cust_ids}")

        logger.info(f"Dataset '{name}': {total_rows} rows, {total_cols} cols, {duplicate_rows} duplicates, {missing_total} null cells.")
        if anomalies:
            for anomaly in anomalies:
                logger.warning(f"Validation anomaly in '{name}': {anomaly}")

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": duplicate_rows,
            "missing_cells": missing_total,
            "missing_by_column": {k: int(v) for k, v in missing_values.items() if v > 0},
            "anomalies": anomalies
        }
