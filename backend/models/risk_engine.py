"""
Customer Risk Stratification & Retention Engine for ECIP.
Categorizes churn probability into 5 risk levels (Very Low, Low, Medium, High, Critical)
and generates personalized retention blueprints.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.RiskEngine")

class ChurnRiskEngine:
    """Stratifies customer churn probabilities and generates tailored retention campaigns."""

    def stratify_risk(self, probability: float) -> str:
        """Categorizes churn probability into 5 risk levels."""
        if probability >= 0.8:
            return "Critical"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        return "Very Low"

    def generate_retention_recommendations(
        self, customer_df: pd.DataFrame, churn_probs: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies risk stratification and attaches personalized retention recommendations.

        Returns:
            Tuple[enriched_predictions_df, high_risk_customers_df]
        """
        logger.info("Stratifying customer churn risk and generating retention recommendations...")
        df = customer_df.copy()
        df["churn_probability"] = churn_probs
        df["risk_level"] = df["churn_probability"].apply(self.stratify_risk)

        recommendations = []
        action_plans = []

        for _, row in df.iterrows():
            prob = row["churn_probability"]
            risk = row["risk_level"]
            recency = row.get("recency_days", 0)
            orders = row.get("total_orders", 1)
            spending = row.get("total_spending", 0)

            if risk == "Critical":
                recs = [
                    "🎯 Urgent Win-Back Discount Code (25% off)",
                    "☎️ Dedicated Customer Success Support Call",
                    "🎁 Free Loyalty Gift with Next Order"
                ]
                plan = "High-priority direct retention outreach within 24 hours."

            elif risk == "High":
                recs = [
                    "✉️ Personalized Re-engagement Email Series",
                    "🚚 Free Express Shipping Voucher",
                    "🏷️ 15% Time-Limited Re-order Coupon"
                ]
                plan = "Targeted automated re-activation sequence within 3 days."

            elif risk == "Medium":
                recs = [
                    "📦 Category Cross-Sell Recommendation Push",
                    "⭐ Loyalty Double-Points Bonus Offer"
                ]
                plan = "Nudge engagement via personalized product recommendations."

            elif risk == "Low":
                recs = [
                    "👑 VIP Tier Progress Notification",
                    "💡 Quarterly Product Catalog Update"
                ]
                plan = "Maintain standard automated email marketing flow."

            else:
                recs = [
                    "🌟 Advocate Review Request & Referral Link",
                    "🚀 Early Access to New Collections"
                ]
                plan = "Brand advocacy and upsell focus."

            recommendations.append(" | ".join(recs))
            action_plans.append(plan)

        df["recommended_retention_actions"] = recommendations
        df["retention_action_plan"] = action_plans

        high_risk_df = df[df["risk_level"].isin(["High", "Critical"])].sort_values(
            by="churn_probability", ascending=False
        )

        logger.info(f"Risk stratification complete: {len(high_risk_df):,} High/Critical risk customers identified.")
        return df, high_risk_df
