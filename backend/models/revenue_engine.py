"""
Revenue Intelligence & Opportunity Engine for ECIP.
Stratifies predicted CLV into Platinum, Gold, Silver, Bronze value tiers and generates
upsell, cross-sell, VIP upgrade, and monthly revenue forecast metrics.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.RevenueEngine")

class RevenueIntelligenceEngine:
    """Classifies predicted CLV value tiers and generates growth opportunity blueprints."""

    def stratify_value_tiers(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Classifies customers into Platinum, Gold, Silver, Bronze, and Low Value tiers."""
        df = predictions_df.copy()
        
        q95 = df["predicted_clv"].quantile(0.95)
        q80 = df["predicted_clv"].quantile(0.80)
        q50 = df["predicted_clv"].quantile(0.50)

        def assign_tier(val: float) -> str:
            if val >= q95:
                return "Platinum (Top 5%)"
            elif val >= q80:
                return "Gold (Top 20%)"
            elif val >= q50:
                return "Silver (Top 50%)"
            elif val > 0:
                return "Bronze"
            return "Low Value"

        df["clv_value_tier"] = df["predicted_clv"].apply(assign_tier)
        return df

    def generate_opportunity_blueprints(
        self, feature_df: pd.DataFrame, predicted_clv: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Engineers revenue opportunity categories and monthly forecasts.

        Returns:
            Tuple[predictions_df, high_value_customers_df, monthly_forecast_df]
        """
        logger.info("Executing Revenue Intelligence & Opportunity engine...")
        df = feature_df.copy()
        df["predicted_clv"] = predicted_clv
        df = self.stratify_value_tiers(df)

        opportunities = []
        recommendations = []

        for _, row in df.iterrows():
            clv = row["predicted_clv"]
            orders = row.get("total_orders", 1)
            spend = row.get("total_spending", 0)
            tier = row["clv_value_tier"]

            if "Platinum" in tier:
                opp = "VIP Upgrade & Concierge Retainer"
                rec = "Offer dedicated account manager, free express shipping, and quarterly preview hampers."

            elif "Gold" in tier and orders >= 2:
                opp = "Subscription & High-Margin Upsell"
                rec = "Pitch annual replenishment subscription and premium product add-ons."

            elif "Silver" in tier and orders == 1:
                opp = "Second Purchase Conversion Campaign"
                rec = "Send triggered 15% discount on complementary product category."

            elif spend < 100 and clv > 300:
                opp = "High-Potential Account Nudge"
                rec = "Cross-sell trending best-sellers to capture latent purchasing capacity."

            else:
                opp = "Standard Re-engagement"
                rec = "Include in bi-weekly promotional emails and seasonal sale alerts."

            opportunities.append(opp)
            recommendations.append(rec)

        df["revenue_opportunity_type"] = opportunities
        df["revenue_recommendation"] = recommendations

        # High Value Customers (Platinum and Gold Tiers)
        high_value_df = df[df["clv_value_tier"].str.contains("Platinum|Gold")].sort_values(
            by="predicted_clv", ascending=False
        )

        # Monthly Revenue Forecast (12-Month Projection Horizon)
        total_projected = df["predicted_clv"].sum()
        monthly_base = total_projected / 12.0
        months = [f"Month {i}" for i in range(1, 13)]
        forecast_vals = [monthly_base * (1.0 + (i * 0.02)) for i in range(12)] # Assume 2% monthly MoM growth trend

        forecast_df = pd.DataFrame({
            "forecast_month": months,
            "projected_revenue": forecast_vals
        })

        logger.info(f"Generated revenue intelligence: ${total_projected:,.2f} total 12-month projected system CLV.")
        return df, high_value_df, forecast_df
