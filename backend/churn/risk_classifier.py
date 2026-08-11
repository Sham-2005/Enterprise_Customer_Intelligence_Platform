"""
Risk Classifier Engine for ECIP Phase 14.
Classifies churn probabilities into 5 Risk Tiers:
- Very Low Risk (0-20%)
- Low Risk (20-40%)
- Medium Risk (40-60%)
- High Risk (60-80%)
- Critical Risk (80-100%)
Computes Risk Distribution, Revenue at Risk by Tier, and Customer Counts per Risk Tier.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.RiskClassifier")

class RiskClassifier:
    """Classifies probabilities into 5 risk tiers and builds breakdown matrices."""

    def stratify_risk_level(self, probability: float) -> str:
        """Classifies a probability float (0.0 to 1.0) into a 5-tier string label."""
        p = float(probability)
        if p >= 0.8:
            return "Critical Risk"
        elif p >= 0.6:
            return "High Risk"
        elif p >= 0.4:
            return "Medium Risk"
        elif p >= 0.2:
            return "Low Risk"
        return "Very Low Risk"

    def get_risk_distribution(
        self,
        churn_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Computes customer counts and revenue volume per risk tier."""
        df = churn_df.copy() if not churn_df.empty else (feature_store_df.copy() if feature_store_df is not None else pd.DataFrame())

        if df.empty:
            return pd.DataFrame({
                "Risk_Tier": ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"],
                "Customer_Count": [0, 0, 0, 0, 0],
                "Total_Revenue": [0.0, 0.0, 0.0, 0.0, 0.0]
            })

        if "churn_probability" not in df.columns:
            if "churn_label" in df.columns:
                df["churn_probability"] = df["churn_label"].astype(float) * 0.75 + 0.10
            else:
                df["churn_probability"] = 0.25

        df["risk_tier"] = df["churn_probability"].apply(self.stratify_risk_level)

        spend_col = "total_spending" if "total_spending" in df.columns else ("historical_clv" if "historical_clv" in df.columns else None)
        cust_col = "customer_unique_id" if "customer_unique_id" in df.columns else df.columns[0]

        agg = df.groupby("risk_tier").agg(
            Customer_Count=(cust_col, "nunique") if cust_col in df.columns else ("risk_tier", "count"),
            Total_Revenue=(spend_col, "sum") if spend_col else ("risk_tier", "count")
        ).reset_index()

        agg.rename(columns={"risk_tier": "Risk_Tier"}, inplace=True)

        # Enforce exact 5-tier ordering
        tier_order = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
        agg["tier_rank"] = agg["Risk_Tier"].map(lambda t: tier_order.index(t) if t in tier_order else 99)
        agg = agg.sort_values("tier_rank").drop(columns=["tier_rank"])

        return agg
