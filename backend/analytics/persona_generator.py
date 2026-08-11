"""
Customer Persona Generator & Business Strategy Engine for ECIP.
Maps unsupervised cluster centroids to intuitive business personas and strategic marketing recommendations.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.PersonaGenerator")

class PersonaGenerator:
    """Automates mapping of machine learning cluster metrics into business personas and retention blueprints."""

    def generate_personas(self, segmented_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates cluster centroids, assigns personas, and maps strategic blueprints.

        Returns:
            Tuple[enriched_segmented_df, personas_summary_df]
        """
        logger.info("Generating automated customer business personas...")

        cluster_summary = segmented_df.groupby("cluster_id").agg(
            customer_count=("customer_unique_id", "count"),
            avg_spending=("total_spending", "mean"),
            avg_orders=("total_orders", "mean"),
            avg_aov=("avg_order_value", "mean"),
            avg_recency=("recency_days", "mean"),
            avg_loyalty=("loyalty_score", "mean"),
            avg_review=("avg_review_score_given", "mean"),
            total_revenue=("total_spending", "sum")
        ).reset_index()

        total_sys_revenue = cluster_summary["total_revenue"].sum()
        cluster_summary["revenue_share_pct"] = (
            cluster_summary["total_revenue"] / max(total_sys_revenue, 1.0)
        ) * 100

        # Rank clusters by spending & orders for dynamic persona assignment
        personas_list = []

        for _, row in cluster_summary.iterrows():
            cid = int(row["cluster_id"])
            spend = row["avg_spending"]
            orders = row["avg_orders"]
            recency = row["avg_recency"]
            loyalty = row["avg_loyalty"]

            # Rule-based Persona Mapping
            if spend > cluster_summary["avg_spending"].median() and recency < cluster_summary["avg_recency"].median():
                title = "VIP Power Buyers"
                desc = "Top tier high-value customers with frequent high-basket purchases."
                rec = "Exclusive VIP previews, dedicated account management, early access to new lines."
                retention = "VIP tier loyalty perks and no-friction direct customer support."
                discount = "Zero discount reliance; focus on luxury service and exclusivity."

            elif orders > cluster_summary["avg_orders"].median() and loyalty > 50:
                title = "Loyal Frequenters"
                desc = "Repeat buyers with consistent order velocity and high brand affinity."
                rec = "Gamified loyalty points, subscription replenishment offers, referral incentives."
                retention = "Automated re-order reminders and double point rewards."
                discount = "10% loyalty cash back on repeat purchases."

            elif recency > cluster_summary["avg_recency"].median() and spend > cluster_summary["avg_spending"].median():
                title = "At-Risk High Rollers"
                desc = "Historically profitable buyers who have shown no activity over the last 90+ days."
                rec = "High-priority win-back campaigns, direct phone/email outreach with special incentives."
                retention = "Personalized win-back survey and steep limited-time revival voucher."
                discount = "20% win-back promo code."

            elif spend < cluster_summary["avg_spending"].median() and orders <= 1.5:
                title = "Inactive One-Timers"
                desc = "Single-purchase buyers with prolonged inactivity and low overall lifetime spend."
                rec = "Automated low-cost re-engagement series, clearance item promotions."
                retention = "Category-based recommendations based on initial purchase."
                discount = "15% welcome-back coupon."

            else:
                title = "Bargain Hunters & Essentials Buyers"
                desc = "Price-sensitive customers purchasing mostly discounted core commodities."
                rec = "Promote bundle deals, flash sales, and low-freight options."
                retention = "Seasonal promotion newsletters and bundle discount codes."
                discount = "Free shipping thresholds and bundle promos."

            personas_list.append({
                "cluster_id": cid,
                "persona_title": title,
                "customer_count": int(row["customer_count"]),
                "revenue_share_pct": round(row["revenue_share_pct"], 2),
                "avg_spending": round(spend, 2),
                "avg_orders": round(orders, 2),
                "avg_recency_days": round(recency, 1),
                "avg_loyalty_index": round(loyalty, 1),
                "business_description": desc,
                "marketing_recommendation": rec,
                "retention_strategy": retention,
                "discount_strategy": discount
            })

        personas_df = pd.DataFrame(personas_list)

        # Merge persona titles back to customer dataframe
        cid_to_title = dict(zip(personas_df["cluster_id"], personas_df["persona_title"]))
        segmented_df["persona_title"] = segmented_df["cluster_id"].map(cid_to_title)

        logger.info(f"Generated {len(personas_df)} business personas successfully.")
        return segmented_df, personas_df
