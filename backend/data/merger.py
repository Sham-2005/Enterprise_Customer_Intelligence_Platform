"""
Data Merger Module for ECIP.
Combines processed Olist tables (Customers, Orders, Items, Products, Sellers, Payments, Reviews)
into a unified relational Master Dataset without duplicate fan-out joins.
"""

from typing import Dict
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("ECIP.DataMerger")

class DataMerger:
    """Merges cleaned relational tables into an analytics-ready master dataset."""

    def merge_datasets(self, cleaned_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Executes strategic join sequence to build the unified Master Dataset."""
        logger.info("Starting relational dataset merging pipeline...")

        customers = cleaned_datasets["customers"]
        orders = cleaned_datasets["orders"]
        order_items = cleaned_datasets["order_items"]
        products = cleaned_datasets["products"]
        payments = cleaned_datasets["payments"]
        reviews = cleaned_datasets["reviews"]
        sellers = cleaned_datasets["sellers"]

        # Step 1: Pre-aggregate Payments by order_id
        logger.info("Aggregating order payments by order_id...")
        payments_agg = payments.groupby("order_id").agg(
            total_payment_value=("payment_value", "sum"),
            max_payment_installments=("payment_installments", "max"),
            preferred_payment_type=("payment_type", lambda x: x.mode()[0] if not x.empty else "unspecified"),
            payment_sequences_count=("payment_sequential", "count")
        ).reset_index()

        # Step 2: Pre-aggregate Reviews by order_id
        logger.info("Aggregating order reviews by order_id...")
        reviews_agg = reviews.groupby("order_id").agg(
            avg_review_score=("review_score", "mean"),
            total_reviews_count=("review_score", "count")
        ).reset_index()

        # Step 3: Join Orders with Customers
        logger.info("Merging Orders with Customers...")
        orders_cust = pd.merge(orders, customers, on="customer_id", how="inner")

        # Step 4: Join with Order Items
        logger.info("Merging with Order Items...")
        master = pd.merge(orders_cust, order_items, on="order_id", how="inner")

        # Step 5: Join with Products
        logger.info("Merging with Products...")
        product_cols = [
            "product_id", "product_category_name_english",
            "product_name_lenght", "product_weight_g"
        ]
        available_pcols = [c for c in product_cols if c in products.columns]
        master = pd.merge(master, products[available_pcols], on="product_id", how="left")

        # Step 6: Join with Sellers
        logger.info("Merging with Sellers...")
        seller_cols = ["seller_id", "seller_city", "seller_state"]
        available_scols = [c for c in seller_cols if c in sellers.columns]
        master = pd.merge(master, sellers[available_scols], on="seller_id", how="left")

        # Step 7: Join with aggregated Payments
        logger.info("Merging aggregated Payments...")
        master = pd.merge(master, payments_agg, on="order_id", how="left")

        # Step 8: Join with aggregated Reviews
        logger.info("Merging aggregated Reviews...")
        master = pd.merge(master, reviews_agg, on="order_id", how="left")

        # Fill missing payment/review values
        master["total_payment_value"] = master["total_payment_value"].fillna(master["price"] + master["freight_value"])
        master["avg_review_score"] = master["avg_review_score"].fillna(3.0)
        master["preferred_payment_type"] = master["preferred_payment_type"].fillna("unspecified")

        logger.info(f"Master Dataset constructed successfully with shape: {master.shape}")
        return master
