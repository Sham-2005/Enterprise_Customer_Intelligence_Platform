"""
Feature Engineering Engine for ECIP.
Computes granular Customer, Product, Order, and Seller features from the relational Master Dataset.
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.FeatureEngineer")

class FeatureEngineer:
    """Computes multidimensional e-commerce feature matrices across Customer, Product, Order, and Seller entities."""

    def compute_all_features(
        self, master_df: pd.DataFrame, snapshot_date: pd.Timestamp = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes features across all business entities.

        Returns:
            Tuple[customer_features_df, product_features_df, order_features_df, seller_features_df]
        """
        logger.info("Starting feature engineering calculations...")

        if snapshot_date is None:
            max_purchase = master_df["order_purchase_timestamp"].max()
            snapshot_date = max_purchase + pd.Timedelta(days=1)
            logger.info(f"Auto-configured reference snapshot_date to {snapshot_date.date()}")

        order_features = self._compute_order_features(master_df)
        customer_features = self._compute_customer_features(master_df, snapshot_date)
        product_features = self._compute_product_features(master_df)
        seller_features = self._compute_seller_features(master_df)

        logger.info("Feature engineering computations completed successfully.")
        return customer_features, product_features, order_features, seller_features

    def _compute_order_features(self, master_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering Order-level features...")
        df = master_df.copy()

        # Delivery Time (days)
        if "order_delivered_customer_date" in df.columns and "order_purchase_timestamp" in df.columns:
            df["delivery_time_days"] = (
                df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
            ).dt.total_seconds() / (24 * 3600)
            df["delivery_time_days"] = df["delivery_time_days"].clip(lower=0)

        # Delivery Delay (actual vs estimated delivery days)
        if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
            df["delivery_delay_days"] = (
                df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
            ).dt.total_seconds() / (24 * 3600)
            df["is_delayed"] = (df["delivery_delay_days"] > 0).astype(int)

        # Shipping Cost Ratio
        total_item_val = df["price"] + df["freight_value"]
        df["shipping_cost_ratio"] = np.where(
            total_item_val > 0, df["freight_value"] / total_item_val, 0.0
        )

        # Date Components & Season
        df["purchase_year"] = df["order_purchase_timestamp"].dt.year
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["purchase_quarter"] = df["order_purchase_timestamp"].dt.quarter
        df["day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        def get_season(month: int) -> str:
            if month in [12, 1, 2]:
                return "Summer"  # Southern Hemisphere (Brazil)
            elif month in [3, 4, 5]:
                return "Autumn"
            elif month in [6, 7, 8]:
                return "Winter"
            return "Spring"

        df["season"] = df["purchase_month"].map(get_season)
        return df

    def _compute_customer_features(self, master_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
        logger.info("Engineering Customer-level features...")
        
        # Group by customer_unique_id for persistent customer identification
        cust_group = master_df.groupby("customer_unique_id")

        customer_features = cust_group.agg(
            total_orders=("order_id", "nunique"),
            total_items_purchased=("order_item_id", "count"),
            total_spending=("price", "sum"),
            total_freight_paid=("freight_value", "sum"),
            avg_order_value=("price", "mean"),
            avg_review_score_given=("avg_review_score", "mean"),
            first_purchase_date=("order_purchase_timestamp", "min"),
            last_purchase_date=("order_purchase_timestamp", "max"),
            distinct_categories_count=("product_category_name_english", "nunique"),
            customer_state=("customer_state", "first"),
            customer_city=("customer_city", "first"),
            preferred_payment_method=("preferred_payment_type", lambda x: x.mode()[0] if not x.empty else "unspecified")
        ).reset_index()

        # Recency (days since last purchase relative to snapshot date)
        customer_features["recency_days"] = (
            snapshot_date - customer_features["last_purchase_date"]
        ).dt.total_seconds() / (24 * 3600)

        # Customer Age / Tenure (days between snapshot and first purchase)
        customer_features["customer_age_days"] = (
            snapshot_date - customer_features["first_purchase_date"]
        ).dt.total_seconds() / (24 * 3600)

        # Lifespan (days between first and last purchase)
        customer_features["purchase_lifespan_days"] = (
            customer_features["last_purchase_date"] - customer_features["first_purchase_date"]
        ).dt.total_seconds() / (24 * 3600)

        # Purchase Frequency (orders per month)
        tenure_months = np.maximum(customer_features["customer_age_days"] / 30.0, 1.0)
        customer_features["purchase_frequency_monthly"] = (
            customer_features["total_orders"] / tenure_months
        )

        # Repeat Purchase Flag
        customer_features["is_repeat_customer"] = (
            customer_features["total_orders"] > 1
        ).astype(int)

        # Average Freight Cost per order
        customer_features["avg_freight_cost"] = (
            customer_features["total_freight_paid"] / np.maximum(customer_features["total_orders"], 1)
        )

        return customer_features

    def _compute_product_features(self, master_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering Product-level features...")
        prod_group = master_df.groupby("product_id")

        product_features = prod_group.agg(
            product_popularity_units=("order_item_id", "count"),
            total_product_revenue=("price", "sum"),
            unique_customers_count=("customer_unique_id", "nunique"),
            avg_product_rating=("avg_review_score", "mean"),
            avg_selling_price=("price", "mean"),
            category=("product_category_name_english", "first")
        ).reset_index()

        return product_features

    def _compute_seller_features(self, master_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering Seller-level features...")
        seller_group = master_df.groupby("seller_id")

        seller_features = seller_group.agg(
            seller_total_revenue=("price", "sum"),
            seller_total_orders=("order_id", "nunique"),
            seller_unique_items_sold=("order_item_id", "count"),
            seller_avg_rating=("avg_review_score", "mean"),
            seller_state=("seller_state", "first"),
            seller_city=("seller_city", "first")
        ).reset_index()

        return seller_features
