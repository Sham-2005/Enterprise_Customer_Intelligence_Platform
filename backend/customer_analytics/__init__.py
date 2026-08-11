"""
Customer Analytics Package for ECIP.
Provides specialized analytics engines for customer KPIs, demographics,
purchasing behavior, loyalty tiers, revenue contribution (Pareto 80/20), and recency activity.
"""

from backend.customer_analytics.customer_kpi_engine import CustomerKPIEngine
from backend.customer_analytics.demographics_engine import CustomerDemographicsEngine
from backend.customer_analytics.behavior_engine import CustomerBehaviorEngine
from backend.customer_analytics.loyalty_engine import CustomerLoyaltyEngine
from backend.customer_analytics.revenue_contribution_engine import RevenueContributionEngine
from backend.customer_analytics.activity_engine import CustomerActivityEngine

__all__ = [
    "CustomerKPIEngine",
    "CustomerDemographicsEngine",
    "CustomerBehaviorEngine",
    "CustomerLoyaltyEngine",
    "RevenueContributionEngine",
    "CustomerActivityEngine"
]
