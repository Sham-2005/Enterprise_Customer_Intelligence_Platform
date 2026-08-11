"""
Customer Value Stratification Engine for ECIP Phase 15.
Classifies customers into 5 Value Tiers:
- Platinum (CLV >= $2,500)
- Gold ($1,200 <= CLV < $2,500)
- Silver ($600 <= CLV < $1,200)
- Bronze ($250 <= CLV < $600)
- Standard (CLV < $250)
Computes Customer Count, Revenue Contribution ($ & %), Average Spending, Average Orders, and Retention Rate per tier.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.ValueClassifier")

class ValueClassifier:
    """Engine for 5-tier customer value classification and matrix summarization."""

    def classify_customer_tier(self, clv_value: float) -> str:
        """Classifies CLV value into one of 5 customer value tiers."""
        v = float(clv_value)
        if v >= 2500.0:
            return "Platinum"
        elif v >= 1200.0:
            return "Gold"
        elif v >= 600.0:
            return "Silver"
        elif v >= 250.0:
            return "Bronze"
        return "Standard"

    def get_value_tier_matrix(
        self,
        clv_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Computes metrics matrix for Platinum, Gold, Silver, Bronze, and Standard tiers."""
        df = clv_df.copy() if not clv_df.empty else (feature_store_df.copy() if feature_store_df is not None else pd.DataFrame())

        if df.empty:
            return pd.DataFrame({
                "Value_Tier": ["Platinum", "Gold", "Silver", "Bronze", "Standard"],
                "Customer_Count": [0, 0, 0, 0, 0],
                "Total_Revenue": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Revenue_Share_Pct": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Avg_Spending": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Avg_Orders": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Retention_Rate": ["0.0%", "0.0%", "0.0%", "0.0%", "0.0%"]
            })

        clv_col = "predicted_clv" if "predicted_clv" in df.columns else ("historical_clv" if "historical_clv" in df.columns else "total_spending")
        if clv_col not in df.columns:
            if "total_spending" in df.columns:
                df["predicted_clv"] = df["total_spending"] * 2.2 + 100.0
            else:
                df["predicted_clv"] = 500.0
            clv_col = "predicted_clv"

        df["value_tier"] = df[clv_col].apply(self.classify_customer_tier)

        spend_col = "total_spending" if "total_spending" in df.columns else clv_col
        orders_col = "total_orders" if "total_orders" in df.columns else clv_col
        cust_col = "customer_unique_id" if "customer_unique_id" in df.columns else df.columns[0]
        churn_col = "churn_label" if "churn_label" in df.columns else None

        tot_sys_rev = df[spend_col].sum() if spend_col in df.columns else 1.0

        agg = df.groupby("value_tier").agg(
            Customer_Count=(cust_col, "nunique") if cust_col in df.columns else ("value_tier", "count"),
            Total_Revenue=(spend_col, "sum") if spend_col in df.columns else ("value_tier", "count"),
            Avg_Spending=(spend_col, "mean") if spend_col in df.columns else ("value_tier", "count"),
            Avg_Orders=(orders_col, "mean") if orders_col in df.columns else ("value_tier", "count"),
            Churn_Rate=(churn_col, "mean") if churn_col else ("value_tier", lambda x: 0.15)
        ).reset_index()

        agg.rename(columns={"value_tier": "Value_Tier"}, inplace=True)
        agg["Revenue_Share_Pct"] = ((agg["Total_Revenue"] / max(tot_sys_rev, 1.0)) * 100.0).round(1)
        agg["Retention_Rate"] = ((1.0 - agg["Churn_Rate"]) * 100.0).round(1).astype(str) + "%"

        tier_order = ["Platinum", "Gold", "Silver", "Bronze", "Standard"]
        agg["tier_rank"] = agg["Value_Tier"].map(lambda t: tier_order.index(t) if t in tier_order else 99)
        agg = agg.sort_values("tier_rank").drop(columns=["tier_rank", "Churn_Rate"])

        return agg
