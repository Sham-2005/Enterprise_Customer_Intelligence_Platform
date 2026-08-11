"""
Customer Purchase Behavior Engine for ECIP Phase 12.
Analyzes Purchase Frequency Distribution, Average Basket Size, AOV, Revenue per Customer,
Product Diversity, and Preferred Payment Methods.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.BehaviorEngine")

class CustomerBehaviorEngine:
    """Engine for customer purchasing behavioral metrics and distributions."""

    def get_purchase_frequency_distribution(self, master_df: pd.DataFrame, feature_store_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Computes distribution of order counts per customer (1 order, 2 orders, 3-5 orders, 6+ orders)."""
        if feature_store_df is not None and not feature_store_df.empty and "total_orders" in feature_store_df.columns:
            counts = feature_store_df["total_orders"]
        elif not master_df.empty and "customer_unique_id" in master_df.columns and "order_id" in master_df.columns:
            counts = master_df.groupby("customer_unique_id")["order_id"].nunique()
        else:
            return pd.DataFrame({"Frequency_Bucket": ["1 Order", "2 Orders", "3-5 Orders", "6+ Orders"], "Customer_Count": [0, 0, 0, 0]})

        b1 = (counts == 1).sum()
        b2 = (counts == 2).sum()
        b3 = ((counts >= 3) & (counts <= 5)).sum()
        b4 = (counts >= 6).sum()

        return pd.DataFrame({
            "Frequency_Bucket": ["1 Order", "2 Orders", "3-5 Orders", "6+ Orders"],
            "Customer_Count": [int(b1), int(b2), int(b3), int(b4)]
        })

    def get_product_diversity_distribution(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """Computes distribution of unique product categories purchased per customer."""
        if master_df.empty or "customer_unique_id" not in master_df.columns or "product_category_name_english" not in master_df.columns:
            return pd.DataFrame({"Diversity_Level": ["1 Category", "2-3 Categories", "4+ Categories"], "Customer_Count": [0, 0, 0]})

        cat_counts = master_df.groupby("customer_unique_id")["product_category_name_english"].nunique()
        c1 = (cat_counts == 1).sum()
        c2 = ((cat_counts >= 2) & (cat_counts <= 3)).sum()
        c3 = (cat_counts >= 4).sum()

        return pd.DataFrame({
            "Diversity_Level": ["1 Category", "2-3 Categories", "4+ Categories"],
            "Customer_Count": [int(c1), int(c2), int(c3)]
        })

    def get_preferred_payment_methods(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """Computes breakdown of preferred payment methods by transaction volume and revenue."""
        if master_df.empty:
            return pd.DataFrame(columns=["Payment_Method", "Transaction_Count", "Total_Revenue"])

        pmt_col = "payment_type" if "payment_type" in master_df.columns else ("preferred_payment_method" if "preferred_payment_method" in master_df.columns else None)
        val_col = "price" if "price" in master_df.columns else "payment_value"

        if not pmt_col:
            return pd.DataFrame(columns=["Payment_Method", "Transaction_Count", "Total_Revenue"])

        agg = master_df.groupby(pmt_col).agg(
            Transaction_Count=(val_col if val_col in master_df.columns else pmt_col, "count"),
            Total_Revenue=(val_col, "sum") if val_col in master_df.columns else (pmt_col, "count")
        ).reset_index()

        agg.rename(columns={pmt_col: "Payment_Method"}, inplace=True)
        agg["Payment_Method"] = agg["Payment_Method"].astype(str).str.replace("_", " ").str.title()
        return agg.sort_values(by="Total_Revenue", ascending=False)

    def get_revenue_per_customer_tiers(self, feature_store_df: Optional[pd.DataFrame] = None, master_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Categorizes customers into spending tiers ($0-$100, $100-$300, $300-$1000, $1000+)."""
        if feature_store_df is not None and not feature_store_df.empty and "total_spending" in feature_store_df.columns:
            spending = feature_store_df["total_spending"]
        elif master_df is not None and not master_df.empty and "price" in master_df.columns and "customer_unique_id" in master_df.columns:
            spending = master_df.groupby("customer_unique_id")["price"].sum()
        else:
            return pd.DataFrame({"Revenue_Tier": ["Low (<$100)", "Medium ($100-$300)", "High ($300-$1K)", "VIP ($1K+)"], "Customer_Count": [0, 0, 0, 0]})

        t1 = (spending < 100).sum()
        t2 = ((spending >= 100) & (spending < 300)).sum()
        t3 = ((spending >= 300) & (spending < 1000)).sum()
        t4 = (spending >= 1000).sum()

        return pd.DataFrame({
            "Revenue_Tier": ["Low (<$100)", "Medium ($100-$300)", "High ($300-$1K)", "VIP ($1K+)"],
            "Customer_Count": [int(t1), int(t2), int(t3), int(t4)]
        })
