"""
Centralized Streamlit Caching Engine for ECIP Dashboard.
Provides high-performance @st.cache_data and @st.cache_resource wrappers for
dataset ingestion, filter options extraction, payload caching, and ML resource management.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import streamlit as st
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.CacheManager")

@st.cache_data(ttl=3600, show_spinner=False)
def load_single_csv_dataset(file_path_str: str) -> pd.DataFrame:
    """Reads and parses a single CSV dataset with Streamlit caching."""
    p = Path(file_path_str)
    if not p.exists():
        logger.warning(f"Cached CSV loader missing file at {file_path_str}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)
        # Parse explicit date columns for maximum performance
        if "order_purchase_timestamp" in df.columns:
            df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
        if "last_purchase_date" in df.columns:
            df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV {file_path_str}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_all_datasets() -> Dict[str, pd.DataFrame]:
    """Loads all core processed datasets using Streamlit @st.cache_data."""
    settings = Settings()
    output_dir = settings.get_path("paths.output_dir")

    files_map = {
        "master_dataset": output_dir / "master_dataset.csv",
        "feature_store": output_dir / "feature_store.csv",
        "customer_metrics": output_dir / "customer_metrics.csv",
        "churn_predictions": output_dir / "customer_churn_predictions.csv",
        "clv_predictions": output_dir / "customer_clv_predictions.csv",
        "processed_customers": output_dir / "processed_customers.csv",
        "rfm_dataset": output_dir / "rfm_dataset.csv"
    }

    datasets = {}
    for key, path in files_map.items():
        if not path.exists() and key == "churn_predictions":
            path = output_dir / "churn_predictions.csv"
        if not path.exists() and key == "clv_predictions":
            path = output_dir / "clv_predictions.csv"

        if path.exists():
            datasets[key] = load_single_csv_dataset(str(path))
        else:
            datasets[key] = pd.DataFrame()

    return datasets


@st.cache_data(ttl=3600, show_spinner=False)
def extract_global_filter_options() -> Dict[str, Any]:
    """
    Extracts global filter options (dates, states, categories, payment methods, segments)
    directly from cached master datasets in < 5ms without running full payload pipelines.
    """
    datasets = get_cached_all_datasets()
    master_df = datasets.get("master_dataset", pd.DataFrame())
    feature_store_df = datasets.get("feature_store", pd.DataFrame())

    date_range = (pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31"))
    if not master_df.empty and "order_purchase_timestamp" in master_df.columns:
        ts = master_df["order_purchase_timestamp"].dropna()
        if not ts.empty:
            date_range = (ts.min(), ts.max())

    states = []
    if not master_df.empty and "customer_state" in master_df.columns:
        states = sorted(list(master_df["customer_state"].dropna().unique()))

    categories = []
    if not master_df.empty and "product_category_name_english" in master_df.columns:
        categories = sorted(list(master_df["product_category_name_english"].dropna().unique()))

    sellers = []
    if not master_df.empty and "seller_id" in master_df.columns:
        sellers = sorted(list(master_df["seller_id"].dropna().unique()))

    payment_methods = []
    if not master_df.empty and "payment_type" in master_df.columns:
        payment_methods = sorted(list(master_df["payment_type"].dropna().unique()))

    customer_segments = []
    if not feature_store_df.empty and "cluster_name" in feature_store_df.columns:
        customer_segments = sorted(list(feature_store_df["cluster_name"].dropna().unique()))

    return {
        "date_range": date_range,
        "states": states,
        "categories": categories,
        "sellers": sellers,
        "payment_methods": payment_methods,
        "customer_segments": customer_segments
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_recommendation_datasets() -> Dict[str, Any]:
    """Loads and caches recommendation module datasets."""
    settings = Settings()
    output_dir = settings.get_path("paths.output_dir")

    files = {
        "customer_recommendations": output_dir / "customer_recommendations.csv",
        "recommended_products": output_dir / "recommended_products.csv",
        "similar_products": output_dir / "similar_products.csv",
        "cross_sell_products": output_dir / "cross_sell_products.csv",
        "upsell_products": output_dir / "upsell_products.csv",
        "trending_products": output_dir / "trending_products.csv",
        "customer_metrics": output_dir / "customer_metrics.csv"
    }

    res = {}
    for key, path in files.items():
        if path.exists():
            res[key] = load_single_csv_dataset(str(path))
        else:
            res[key] = pd.DataFrame()

    all_d = get_cached_all_datasets()
    res["master_dataset"] = all_d.get("master_dataset", pd.DataFrame())
    res["feature_store"] = all_d.get("feature_store", pd.DataFrame())
    res["clv_predictions"] = all_d.get("clv_predictions", pd.DataFrame())
    res["churn_predictions"] = all_d.get("churn_predictions", pd.DataFrame())
    return res


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_mba_datasets() -> Dict[str, Any]:
    """Loads and caches Market Basket Analysis association rules and bundle datasets."""
    settings = Settings()
    output_dir = settings.get_path("paths.output_dir")

    files = {
        "association_rules": output_dir / "association_rules.csv",
        "product_bundles": output_dir / "product_bundles.csv",
        "cross_sell_recommendations": output_dir / "cross_sell_recommendations.csv",
        "basket_statistics": output_dir / "basket_statistics.csv"
    }

    res = {}
    for key, path in files.items():
        if path.exists():
            res[key] = load_single_csv_dataset(str(path))
        else:
            res[key] = pd.DataFrame()

    all_d = get_cached_all_datasets()
    res["master_dataset"] = all_d.get("master_dataset", pd.DataFrame())
    res["customer_segments"] = all_d.get("feature_store", pd.DataFrame())
    res["customer_metrics"] = all_d.get("customer_metrics", pd.DataFrame())
    return res


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_mlops_data() -> Dict[str, Any]:
    """Loads and caches MLOps registry, drift, and telemetry metrics."""
    settings = Settings()
    models_dir = settings.get_path("paths.models_dir")
    reports_dir = settings.get_path("paths.reports_dir")

    registry_file = models_dir / "model_registry.json"
    experiments_file = models_dir / "experiments_log.json"
    drift_file = reports_dir / "drift_report.json"

    registered = {}
    if registry_file.exists():
        try:
            import json
            with open(registry_file, "r", encoding="utf-8") as f:
                raw_reg = json.load(f)
                if isinstance(raw_reg, dict) and "registered_models" in raw_reg:
                    registered = raw_reg["registered_models"]
                elif isinstance(raw_reg, dict):
                    registered = {k: v for k, v in raw_reg.items() if isinstance(v, dict)}
        except Exception:
            pass

    experiments = []
    if experiments_file.exists():
        try:
            import json
            with open(experiments_file, "r", encoding="utf-8") as f:
                experiments = json.load(f)
        except Exception:
            pass

    drift_report = {}
    if drift_file.exists():
        try:
            import json
            with open(drift_file, "r", encoding="utf-8") as f:
                drift_report = json.load(f)
        except Exception:
            pass

    return {
        "registered_models": registered,
        "experiments": experiments,
        "drift_report": drift_report
    }


