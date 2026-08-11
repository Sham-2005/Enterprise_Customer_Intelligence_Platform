"""
Data Cleaning & Preprocessor Module for ECIP.
Handles date parsing, string standardization, handling missing values, category translations,
and dropping invalid/corrupted records.
"""

from typing import Dict
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.DataPreprocessor")

class DataPreprocessor:
    """Preprocesses and cleans raw Olist dataframes into normalized tables."""

    def clean_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Runs cleaning steps across all loaded dataframes."""
        logger.info("Executing dataset cleaning and preprocessing steps...")
        cleaned = {}

        cleaned["customers"] = self._clean_customers(datasets["customers"].copy())
        cleaned["orders"] = self._clean_orders(datasets["orders"].copy())
        cleaned["order_items"] = self._clean_order_items(datasets["order_items"].copy())
        cleaned["products"] = self._clean_products(
            datasets["products"].copy(),
            datasets.get("category_translation", None)
        )
        cleaned["payments"] = self._clean_payments(datasets["payments"].copy())
        cleaned["reviews"] = self._clean_reviews(datasets["reviews"].copy())
        cleaned["sellers"] = self._clean_sellers(datasets["sellers"].copy())

        logger.info("Dataset cleaning and preprocessing completed successfully.")
        return cleaned

    def _clean_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["customer_id"])
        df["customer_city"] = df["customer_city"].astype(str).str.strip().str.title()
        df["customer_state"] = df["customer_state"].astype(str).str.strip().str.upper()
        return df

    def _clean_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["order_id"])
        
        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Standardize order_status
        if "order_status" in df.columns:
            df["order_status"] = df["order_status"].astype(str).str.strip().str.lower()

        return df

    def _clean_order_items(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        
        # Remove negative prices or freight
        df = df[df["price"] >= 0]
        df = df[df["freight_value"] >= 0]

        if "shipping_limit_date" in df.columns:
            df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")

        return df

    def _clean_products(self, df: pd.DataFrame, translation_df: pd.DataFrame = None) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["product_id"])

        if translation_df is not None and not translation_df.empty:
            logger.info("Translating product categories from Portuguese to English...")
            trans_map = dict(zip(translation_df["product_category_name"], translation_df["product_category_name_english"]))
            df["product_category_name_english"] = df["product_category_name"].map(trans_map)
            df["product_category_name_english"] = df["product_category_name_english"].fillna(
                df["product_category_name"].fillna("unspecified")
            )
        else:
            df["product_category_name_english"] = df["product_category_name"].fillna("unspecified")

        # Fill missing numerical measurements with median
        num_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _clean_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["payment_value"] >= 0]
        df["payment_type"] = df["payment_type"].astype(str).str.strip().str.lower()
        return df

    def _clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        if "review_score" in df.columns:
            df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce").fillna(3.0)

        date_cols = ["review_creation_date", "review_answer_timestamp"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def _clean_sellers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["seller_id"])
        df["seller_city"] = df["seller_city"].astype(str).str.strip().str.title()
        df["seller_state"] = df["seller_state"].astype(str).str.strip().str.upper()
        return df
