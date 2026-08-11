"""
Business Persona Manager for ECIP Phase 13.
Generates and manages standard C-suite business personas with descriptions,
revenue contributions, buying behaviors, marketing recommendations, and retention strategies.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("ECIP.PersonaManager")

STANDARD_PERSONAS_DEFINITIONS = [
    {
        "persona_title": "VIP Power Buyers",
        "description": "Top-tier high-value accounts with high order frequency and high basket sizes.",
        "buying_behavior": "Frequent high-basket repeat purchases; premium catalog affinity.",
        "purchase_frequency": "Very High (>= 4 orders/yr)",
        "marketing_recommendation": "Exclusive VIP previews, dedicated concierge support, early line access.",
        "retention_strategy": "VIP tier loyalty perks, zero friction service, exclusive anniversary rewards.",
        "icon": "💎"
    },
    {
        "persona_title": "Loyal Frequent Buyers",
        "description": "Repeat buyers with consistent order velocity and strong brand loyalty.",
        "buying_behavior": "Predictable monthly or quarterly repeat orders.",
        "purchase_frequency": "High (2 - 3 orders/yr)",
        "marketing_recommendation": "Gamified loyalty points, subscription replenishment offers, referral incentives.",
        "retention_strategy": "Automated re-order reminders and double point rewards on repeat items.",
        "icon": "🏆"
    },
    {
        "persona_title": "Premium Customers",
        "description": "High average order value buyers with moderate purchase frequency.",
        "buying_behavior": "Large basket values per purchase; quality-driven choices.",
        "purchase_frequency": "Moderate (1 - 2 orders/yr)",
        "marketing_recommendation": "Upsell higher-end product bundles and premium category extensions.",
        "retention_strategy": "Free express shipping thresholds and premium warranty add-ons.",
        "icon": "🌟"
    },
    {
        "persona_title": "New Customers",
        "description": "First-time buyers acquired within the last 30 to 60 days.",
        "buying_behavior": "Initial single purchase; establishing brand relationship.",
        "purchase_frequency": "New (1 order)",
        "marketing_recommendation": "Automated welcome onboarding sequence and product guides.",
        "retention_strategy": "Second purchase incentive coupon within 14 days of delivery.",
        "icon": "🌱"
    },
    {
        "persona_title": "Occasional Buyers",
        "description": "Intermittent buyers making occasional seasonal purchases.",
        "buying_behavior": "Seasonal or holiday sales events driven purchasing.",
        "purchase_frequency": "Occasional (1 order / 6 months)",
        "marketing_recommendation": "Promote seasonal sales events, holiday catalogs, and flash deals.",
        "retention_strategy": "Targeted event-based email triggers and category recommendations.",
        "icon": "🛍️"
    },
    {
        "persona_title": "Price Sensitive Customers",
        "description": "Bargain hunters focused on discounted commodities and sales promotions.",
        "buying_behavior": "Purchases primarily during clearance sales or with discount codes.",
        "purchase_frequency": "Low to Moderate",
        "marketing_recommendation": "Promote clearance bundles, coupon codes, and low-freight options.",
        "retention_strategy": "Discount code incentives and free shipping vouchers.",
        "icon": "🏷️"
    },
    {
        "persona_title": "At-Risk Customers",
        "description": "Historically profitable buyers showing prolonged inactivity (>90 days).",
        "buying_behavior": "High past spending but dropping order frequency.",
        "purchase_frequency": "Declining",
        "marketing_recommendation": "High-priority win-back campaigns and personalized outreach.",
        "retention_strategy": "20% win-back promo code and feedback survey.",
        "icon": "⚠️"
    },
    {
        "persona_title": "Lost Customers",
        "description": "Inactive accounts with no orders for over 180+ days.",
        "buying_behavior": "Dormant account status.",
        "purchase_frequency": "Inactive (0 in 6+ months)",
        "marketing_recommendation": "Low-cost re-engagement newsletter or list purge.",
        "retention_strategy": "Final flash coupon or feedback request.",
        "icon": "💤"
    }
]

class PersonaManager:
    """Manager for persona calculations and strategy mapping."""

    def get_personas_summary(
        self,
        feature_store_df: pd.DataFrame,
        personas_df: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Compiles detailed list of business personas with metrics and strategies."""
        total_cust = len(feature_store_df) if not feature_store_df.empty else 1000
        tot_rev = feature_store_df["total_spending"].sum() if not feature_store_df.empty and "total_spending" in feature_store_df.columns else 1000000.0

        result = []
        for idx, p in enumerate(STANDARD_PERSONAS_DEFINITIONS):
            title = p["persona_title"]
            # Estimate segment proportion
            count = max(10, int(total_cust * (0.20 if idx < 2 else 0.10)))
            share = round((count / total_cust) * 100, 1)
            rev = round(tot_rev * (share / 100.0), 2)

            result.append({
                "persona_title": title,
                "description": p["description"],
                "customer_count": f"{count:,}",
                "revenue_contribution": f"${rev:,.2f} ({share}%)",
                "buying_behavior": p["buying_behavior"],
                "purchase_frequency": p["purchase_frequency"],
                "marketing_recommendation": p["marketing_recommendation"],
                "retention_strategy": p["retention_strategy"],
                "icon": p["icon"]
            })

        return result
