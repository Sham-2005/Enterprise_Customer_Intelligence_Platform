"""
Customer Activity & Recency Analysis Engine for ECIP Phase 12.
Analyzes Recency Distribution, Purchase Intervals, Active Days,
and extracts rosters for Recently Active Customers and Dormant Customers (>90 days inactive).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.ActivityEngine")

class CustomerActivityEngine:
    """Engine for recency buckets, customer lifespan, and active/dormant roster extraction."""

    def get_recency_distribution(
        self,
        feature_store_df: Optional[pd.DataFrame] = None,
        master_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Computes customer recency distribution buckets (0-30d, 31-60d, 61-90d, 91-180d, 180+d)."""
        recency_series = None
        if feature_store_df is not None and not feature_store_df.empty and "recency_days" in feature_store_df.columns:
            recency_series = feature_store_df["recency_days"].dropna()
        elif master_df is not None and not master_df.empty and "order_purchase_timestamp" in master_df.columns:
            date_col = "order_purchase_timestamp"
            df_m = master_df.copy()
            df_m[date_col] = pd.to_datetime(df_m[date_col], errors="coerce")
            max_date = df_m[date_col].max()
            if pd.notna(max_date) and "customer_unique_id" in df_m.columns:
                last_purchase = df_m.groupby("customer_unique_id")[date_col].max()
                recency_series = (max_date - last_purchase).dt.days

        if recency_series is None or len(recency_series) == 0:
            return pd.DataFrame({
                "Recency_Bucket": ["0-30 Days", "31-60 Days", "61-90 Days", "91-180 Days", "180+ Days"],
                "Customer_Count": [0, 0, 0, 0, 0]
            })

        r1 = (recency_series <= 30).sum()
        r2 = ((recency_series > 30) & (recency_series <= 60)).sum()
        r3 = ((recency_series > 60) & (recency_series <= 90)).sum()
        r4 = ((recency_series > 90) & (recency_series <= 180)).sum()
        r5 = (recency_series > 180).sum()

        return pd.DataFrame({
            "Recency_Bucket": ["0-30 Days", "31-60 Days", "61-90 Days", "91-180 Days", "180+ Days"],
            "Customer_Count": [int(r1), int(r2), int(r3), int(r4), int(r5)]
        })

    def get_recently_active_customers(
        self,
        master_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None,
        top_n: int = 50
    ) -> pd.DataFrame:
        """Extracts roster of top N recently active customer accounts."""
        if master_df.empty and (feature_store_df is None or feature_store_df.empty):
            return pd.DataFrame(columns=["Customer_ID", "Last_Purchase_Date", "Recency_Days", "Total_Spending", "Status"])

        if not master_df.empty and "customer_unique_id" in master_df.columns and "order_purchase_timestamp" in master_df.columns:
            df_m = master_df.copy()
            df_m["order_purchase_timestamp"] = pd.to_datetime(df_m["order_purchase_timestamp"], errors="coerce")
            max_date = df_m["order_purchase_timestamp"].max()
            val_col = "price" if "price" in df_m.columns else "payment_value"

            agg = df_m.groupby("customer_unique_id").agg(
                Last_Purchase_Date=("order_purchase_timestamp", "max"),
                Total_Spending=(val_col, "sum"),
                Total_Orders=("order_id", "nunique") if "order_id" in df_m.columns else (val_col, "count")
            ).reset_index()

            agg["Recency_Days"] = (max_date - agg["Last_Purchase_Date"]).dt.days
            agg.rename(columns={"customer_unique_id": "Customer_ID"}, inplace=True)
            agg["Status"] = "Active"
            agg["Last_Purchase_Date"] = agg["Last_Purchase_Date"].dt.strftime("%Y-%m-%d")
            return agg.sort_values(by="Recency_Days", ascending=True).head(top_n)

        elif feature_store_df is not None and not feature_store_df.empty:
            df = feature_store_df.copy()
            id_col = "customer_unique_id" if "customer_unique_id" in df.columns else df.columns[0]
            rec_col = "recency_days" if "recency_days" in df.columns else df.columns[1]
            spend_col = "total_spending" if "total_spending" in df.columns else rec_col

            agg = df[[id_col, rec_col, spend_col]].copy()
            agg.columns = ["Customer_ID", "Recency_Days", "Total_Spending"]
            agg["Last_Purchase_Date"] = "Recent"
            agg["Status"] = "Active"
            return agg.sort_values(by="Recency_Days", ascending=True).head(top_n)

        return pd.DataFrame(columns=["Customer_ID", "Last_Purchase_Date", "Recency_Days", "Total_Spending", "Status"])

    def get_dormant_customers(
        self,
        master_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None,
        top_n: int = 50
    ) -> pd.DataFrame:
        """Extracts roster of dormant customer accounts (inactivity > 90 days)."""
        if feature_store_df is not None and not feature_store_df.empty and "recency_days" in feature_store_df.columns:
            dormant = feature_store_df[feature_store_df["recency_days"] > 90].copy()
            if not dormant.empty:
                id_col = "customer_unique_id" if "customer_unique_id" in dormant.columns else dormant.columns[0]
                spend_col = "total_spending" if "total_spending" in dormant.columns else "recency_days"
                orders_col = "total_orders" if "total_orders" in dormant.columns else spend_col

                agg = dormant[[id_col, "recency_days", spend_col, orders_col]].copy()
                agg.columns = ["Customer_ID", "Recency_Days", "Total_Spending", "Total_Orders"]
                agg["Status"] = "Dormant (>90d)"
                return agg.sort_values(by="Recency_Days", ascending=False).head(top_n)

        active_df = self.get_recently_active_customers(master_df, feature_store_df, top_n=500)
        if not active_df.empty and "Recency_Days" in active_df.columns:
            dormant = active_df[active_df["Recency_Days"] > 90].copy()
            dormant["Status"] = "Dormant (>90d)"
            return dormant.sort_values(by="Recency_Days", ascending=False).head(top_n)

        return pd.DataFrame(columns=["Customer_ID", "Recency_Days", "Total_Spending", "Total_Orders", "Status"])
