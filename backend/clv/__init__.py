"""
Customer Lifetime Value (CLV) & Revenue Intelligence Package for ECIP.
Provides specialized engines for CLV KPIs, 5-Tier Customer Value Stratification,
Revenue Forecasting, Opportunity Intelligence Recommendations, and SHAP Explainable AI for Regression.
"""

from backend.clv.clv_kpi_engine import CLVKPIEngine
from backend.clv.value_classifier import ValueClassifier
from backend.clv.opportunity_intelligence import OpportunityIntelligenceEngine
from backend.clv.clv_explainability_engine import CLVExplainabilityEngine
from backend.clv.revenue_forecast_engine import RevenueForecastEngine

__all__ = [
    "CLVKPIEngine",
    "ValueClassifier",
    "OpportunityIntelligenceEngine",
    "CLVExplainabilityEngine",
    "RevenueForecastEngine"
]
