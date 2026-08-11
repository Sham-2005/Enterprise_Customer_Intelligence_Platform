"""
RFM Analytics & Customer Intelligence Engine for ECIP.
Computes Recency, Frequency, Monetary metrics, RFM quintiles, segment assignments,
loyalty index, priority rankings, and actionable marketing recommendations.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.RFMAnalyzer")

class RFMAnalyzer:
    """Computes advanced RFM analytics, segment classification, and marketing strategy recommendations."""

    def analyze_rfm(self, feature_store_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes complete RFM calculation, segment assignment, and strategy mapping.

        Returns:
            Tuple[rfm_scores_df, segment_summary_df]
        """
        logger.info("Executing RFM intelligence analysis...")
        df = feature_store_df.copy()

        df = self._compute_rfm_quintiles(df)
        df = self._assign_rfm_segments(df)
        df = self._compute_priority_ranking(df)

        segment_summary = self._build_segment_summary(df)

        logger.info("RFM intelligence analysis completed successfully.")
        return df, segment_summary

    def _compute_rfm_quintiles(self, df: pd.DataFrame) -> pd.DataFrame:
        # Recency Score (1-5, lower recency_days gets higher score)
        df["r_score"] = pd.qcut(
            df["recency_days"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
        ).astype(int)

        # Frequency Score (1-5)
        try:
            df["f_score"] = pd.qcut(
                df["total_orders"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
            ).astype(int)
        except Exception:
            df["f_score"] = np.where(df["total_orders"] > 1, 5, 1)

        # Monetary Score (1-5)
        df["m_score"] = pd.qcut(
            df["total_spending"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
        ).astype(int)

        df["rfm_combined_score"] = (
            df["r_score"].astype(str) + df["f_score"].astype(str) + df["m_score"].astype(str)
        )
        return df

    def _assign_rfm_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_rfm_segment(row) -> str:
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3 and m >= 3:
                return "Loyal Customers"
            elif r >= 3 and f >= 1 and m >= 3:
                return "Potential Loyalists"
            elif r >= 4 and f == 1:
                return "New Customers"
            elif r == 3 and f == 1:
                return "Promising"
            elif r == 3 and f >= 2:
                return "Need Attention"
            elif r == 2 and f >= 2:
                return "At Risk"
            elif r <= 2 and f >= 4:
                return "Can't Lose Them"
            elif r == 2 and f == 1:
                return "Hibernating"
            else:
                return "Lost Customers"

        df["rfm_segment_label"] = df.apply(get_rfm_segment, axis=1)
        return df

    def _compute_priority_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        # Priority Score (0 - 100) weighted heavily on monetary value and frequency
        norm_m = (df["total_spending"] - df["total_spending"].min()) / np.maximum(
            df["total_spending"].max() - df["total_spending"].min(), 1.0
        )
        norm_r = (df["recency_days"].max() - df["recency_days"]) / np.maximum(
            df["recency_days"].max() - df["recency_days"].min(), 1.0
        )
        norm_f = (df["total_orders"] - df["total_orders"].min()) / np.maximum(
            df["total_orders"].max() - df["total_orders"].min(), 1.0
        )

        df["customer_priority_score"] = (
            (norm_m * 0.5 + norm_r * 0.3 + norm_f * 0.2) * 100
        ).round(2)

        df["priority_rank"] = df["customer_priority_score"].rank(ascending=False, method="min").astype(int)
        return df

    def _build_segment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.groupby("rfm_segment_label").agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_orders=("total_orders", "mean"),
            avg_spending=("total_spending", "mean"),
            total_revenue=("total_spending", "sum")
        ).reset_index()

        # Add Marketing Recommendations
        recommendations = {
            "Champions": {
                "objective": "Reward & Retain",
                "campaign": "VIP Exclusives & Early Access",
                "discount_strategy": "No aggressive discounts required",
                "channel": "Personalized VIP Concierge Email",
                "cross_sell": "New High-End Collections",
                "upsell": "Premium Bundles"
            },
            "Loyal Customers": {
                "objective": "Upsell & Advocacy",
                "campaign": "Loyalty Tier Progression & Reviews",
                "discount_strategy": "Reward Points / Loyalty Cash",
                "channel": "Targeted Product Recommendation Emails",
                "cross_sell": "Complementary Accessories",
                "upsell": "Higher Volume Packs"
            },
            "Potential Loyalists": {
                "objective": "Increase Order Frequency",
                "campaign": "Membership Perks & Free Shipping",
                "discount_strategy": "10% off next order within 14 days",
                "channel": "Triggered Post-Purchase Nudge",
                "cross_sell": "Trending Best-Sellers",
                "upsell": "Subscription Options"
            },
            "New Customers": {
                "objective": "Onboarding & Habit Building",
                "campaign": "Welcome Series & Product Guides",
                "discount_strategy": "15% Welcome Discount Code",
                "channel": "Automated Onboarding Sequence",
                "cross_sell": "Starter Kits",
                "upsell": "Popular Bundles"
            },
            "Promising": {
                "objective": "Brand Engagement",
                "campaign": "Limited Time Seasonal Offers",
                "discount_strategy": "Free Shipping Voucher",
                "channel": "Category Spotlights",
                "cross_sell": "Related Sub-Categories",
                "upsell": "Slightly Higher AOV Items"
            },
            "Need Attention": {
                "objective": "Re-activate Before Dropoff",
                "campaign": "Re-engagement 'We Miss You' Push",
                "discount_strategy": "15% Time-Limited Promo",
                "channel": "SMS & Email Dual Push",
                "cross_sell": "New Arrivals",
                "upsell": "Multi-Packs"
            },
            "At Risk": {
                "objective": "Win-Back High Value",
                "campaign": "Exclusive Win-Back Special",
                "discount_strategy": "20% Win-Back Discount",
                "channel": "High-Priority Direct Email",
                "cross_sell": "Top Rated Essentials",
                "upsell": "Re-stock Offers"
            },
            "Can't Lose Them": {
                "objective": "Aggressive Retention",
                "campaign": "Direct Outreach & Heavy Incentive",
                "discount_strategy": "25% Heavy Discount + Free Gift",
                "channel": "Dedicated Account Manager / VIP Call",
                "cross_sell": "Flagship Products",
                "upsell": "Custom Contracts"
            },
            "Hibernating": {
                "objective": "Re-ignite Interest",
                "campaign": "Brand Comeback Story & New Features",
                "discount_strategy": "Steep Clearance Discounts",
                "channel": "Batch Re-engagement Newsletter",
                "cross_sell": "Value Packs",
                "upsell": "N/A"
            },
            "Lost Customers": {
                "objective": "Low-Cost Survey / Purge",
                "campaign": "Feedback Survey & Last Chance Offer",
                "discount_strategy": "Final 30% Flash Coupon",
                "channel": "Automated Low-Frequency Campaign",
                "cross_sell": "N/A",
                "upsell": "N/A"
            }
        }

        # Map recommendations
        for key in ["objective", "campaign", "discount_strategy", "channel", "cross_sell", "upsell"]:
            summary[key] = summary["rfm_segment_label"].map(
                lambda s: recommendations.get(s, {}).get(key, "General Promotion")
            )

        return summary
