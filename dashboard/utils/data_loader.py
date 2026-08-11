"""
Dashboard Data Loader & Caching Utility for ECIP.
Loads processed Feature Store and Master datasets using Streamlit caching.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import streamlit as st
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.DashboardDataLoader")

@st.cache_data(ttl=3600)
def load_dashboard_datasets() -> Dict[str, pd.DataFrame]:
    """Loads and caches processed CSV datasets from the output directory."""
    settings = Settings()
    output_dir = settings.get_path("paths.output_dir")

    datasets = {}
    files_map = {
        "master": output_dir / "master_dataset.csv",
        "feature_store": output_dir / "feature_store.csv",
        "processed_customers": output_dir / "processed_customers.csv",
        "rfm": output_dir / "rfm_dataset.csv",
        "customer_metrics": output_dir / "customer_metrics.csv"
    }

    for key, path in files_map.items():
        if path.exists():
            df = pd.read_csv(path)
            # Parse dates if present
            for col in df.columns:
                if "date" in col or "timestamp" in col:
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    except Exception:
                        pass
            datasets[key] = df
            logger.info(f"Dashboard cached '{key}' dataset (Shape: {df.shape})")
        else:
            logger.warning(f"Dataset file missing at {path}")

    return datasets


def filter_dataset(
    df: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    states: Optional[list] = None,
    categories: Optional[list] = None,
    sellers: Optional[list] = None
) -> pd.DataFrame:
    """Applies sidebar filter conditions to a target dataframe."""
    filtered_df = df.copy()

    # Date Range Filter
    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if "order_purchase_timestamp" in filtered_df.columns:
            filtered_df["order_purchase_timestamp"] = pd.to_datetime(filtered_df["order_purchase_timestamp"])
            filtered_df = filtered_df[
                (filtered_df["order_purchase_timestamp"] >= start_date) &
                (filtered_df["order_purchase_timestamp"] <= end_date)
            ]
        elif "last_purchase_date" in filtered_df.columns:
            filtered_df["last_purchase_date"] = pd.to_datetime(filtered_df["last_purchase_date"])
            filtered_df = filtered_df[
                (filtered_df["last_purchase_date"] >= start_date) &
                (filtered_df["last_purchase_date"] <= end_date)
            ]

    # State Filter
    if states:
        if "customer_state" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["customer_state"].isin(states)]
        elif "seller_state" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["seller_state"].isin(states)]

    # Product Category Filter
    if categories and "product_category_name_english" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["product_category_name_english"].isin(categories)]

    # Seller Filter
    if sellers and "seller_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["seller_id"].isin(sellers)]

    return filtered_df
