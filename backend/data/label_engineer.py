"""
Derived Label & Target Generator for ECIP.
Engineers configurable Churn labels, RFM scores, Customer Lifetime Value (CLV), Loyalty Scores,
Spending Tiers, and Segment Labels.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.LabelEngineer")

class LabelEngineer:
    """Computes configurable target labels, segmentation metrics, and business tier classifications."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.inactivity_threshold = self.settings.get("analytics.churn.inactivity_threshold_days", 90)

    def generate_all_labels(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """Applies churn labeling, RFM segmentation, CLV, loyalty scoring, and spending tiers."""
        logger.info("Starting derived machine learning label generation...")
        df = customer_df.copy()

        df = self._generate_churn_label(df)
        df = self._generate_rfm_scores(df)
        df = self._generate_clv_metrics(df)
        df = self._generate_loyalty_and_tiers(df)

        logger.info("Derived labels and segments generated successfully.")
        return df

    def _generate_churn_label(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating Churn label (Inactivity Threshold = {self.inactivity_threshold} days)...")
        df["churn_label"] = (df["recency_days"] > self.inactivity_threshold).astype(int)
        churn_rate = df["churn_label"].mean() * 100
        logger.info(f"Churn label generated: {churn_rate:.2f}% active churn rate detected.")
        return df

    def _generate_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing RFM quintiles and segment labels...")
        
        # Recency Score (1-5, lower recency_days gets higher score)
        df["recency_score"] = pd.qcut(
            df["recency_days"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
        ).astype(int)

        # Frequency Score (1-5, handles low variance in frequency gracefully)
        try:
            df["frequency_score"] = pd.qcut(
                df["total_orders"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
            ).astype(int)
        except Exception:
            df["frequency_score"] = np.where(df["total_orders"] > 1, 5, 1)

        # Monetary Score (1-5, higher total_spending gets higher score)
        df["monetary_score"] = pd.qcut(
            df["total_spending"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # RFM Composite Score
        df["rfm_combined"] = (
            df["recency_score"].astype(str) + 
            df["frequency_score"].astype(str) + 
            df["monetary_score"].astype(str)
        )

        def assign_rfm_segment(row) -> str:
            r, f, m = row["recency_score"], row["frequency_score"], row["monetary_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 3 and f < 3 and m >= 3:
                return "Potential Loyalists"
            elif r == 3 and f < 3:
                return "Promising / Recent"
            elif r == 2 and f >= 2:
                return "At Risk"
            elif r <= 2 and f >= 4:
                return "Cant Lose Them"
            elif r == 2 and f <= 2:
                return "Hibernating"
            else:
                return "Lost / Inactive"

        df["rfm_segment"] = df.apply(assign_rfm_segment, axis=1)
        return df

    def _generate_clv_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Customer Lifetime Value (CLV) baseline metrics...")
        df["historical_clv"] = df["total_spending"] + df["total_freight_paid"]
        df["clv_per_order"] = df["historical_clv"] / np.maximum(df["total_orders"], 1)
        return df

    def _generate_loyalty_and_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Loyalty Index and Spending Tiers...")
        
        # Spending Tier
        spend_quantiles = df["total_spending"].quantile([0.5, 0.8, 0.95]).values
        def get_tier(spend: float) -> str:
            if spend >= spend_quantiles[2]:
                return "Tier 4 - Platinum VIP"
            elif spend >= spend_quantiles[1]:
                return "Tier 3 - Gold"
            elif spend >= spend_quantiles[0]:
                return "Tier 2 - Silver"
            return "Tier 1 - Bronze"

        df["spending_tier"] = df["total_spending"].apply(get_tier)

        # Composite Loyalty Score (0 to 100)
        norm_freq = (df["total_orders"] - df["total_orders"].min()) / np.maximum(
            df["total_orders"].max() - df["total_orders"].min(), 1
        )
        norm_spend = (df["total_spending"] - df["total_spending"].min()) / np.maximum(
            df["total_spending"].max() - df["total_spending"].min(), 1
        )
        norm_rating = df["avg_review_score_given"] / 5.0

        df["loyalty_score"] = (
            (norm_freq * 0.4 + norm_spend * 0.4 + norm_rating * 0.2) * 100
        ).round(2)

        return df
