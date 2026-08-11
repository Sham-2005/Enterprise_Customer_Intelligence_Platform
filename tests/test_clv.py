"""
Unit & Integration Test Suite for Phase 15 - Customer Lifetime Value (CLV) & Revenue Intelligence Dashboard.
Tests 8 CLV KPIs, 5-Tier Value Stratification, Opportunity Intelligence Recommendations,
SHAP Explainable AI for CLV Regression, Revenue Forecasting, Top 100 Leaderboard, Pareto Concentration,
Search Engine, Filters, Exports, and Graceful Error Handling.
"""

import pytest
import pandas as pd
import numpy as np
from backend.clv.clv_kpi_engine import CLVKPIEngine
from backend.clv.value_classifier import ValueClassifier
from backend.clv.opportunity_intelligence import OpportunityIntelligenceEngine
from backend.clv.clv_explainability_engine import CLVExplainabilityEngine
from backend.clv.revenue_forecast_engine import RevenueForecastEngine
from backend.services.clv_service import CLVService

@pytest.fixture
def dummy_clv_df():
    """Generates synthetic customer CLV prediction dataset."""
    data = {
        "customer_unique_id": [f"CUST_{i:03d}" for i in range(30)],
        "predicted_clv": np.random.uniform(100, 5000, size=30),
        "total_spending": np.random.uniform(50, 3000, size=30),
        "total_orders": np.random.randint(1, 10, size=30),
        "avg_order_value": np.random.uniform(50, 400, size=30),
        "recency_days": np.random.randint(1, 180, size=30),
        "historical_clv": np.random.uniform(100, 5000, size=30),
        "loyalty_score": np.random.uniform(10, 100, size=30),
        "avg_review_score_given": np.random.uniform(1.0, 5.0, size=30),
        "rfm_segment": np.random.choice(["Champions", "Loyal Customers", "At Risk", "Hibernating"], size=30),
        "value_tier": np.random.choice(["Platinum", "Gold", "Silver", "Bronze", "Standard"], size=30)
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_master_df():
    """Generates synthetic master transaction dataset."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = {
        "order_id": [f"ORD_{i:04d}" for i in range(100)],
        "customer_unique_id": [f"CUST_{i%30:03d}" for i in range(100)],
        "price": np.random.uniform(20, 300, size=100),
        "order_purchase_timestamp": dates,
        "customer_state": np.random.choice(["SP", "RJ", "MG"], size=100),
        "product_category_name_english": np.random.choice(["bed_bath_table", "health_beauty"], size=100),
        "payment_type": np.random.choice(["credit_card", "boleto"], size=100)
    }
    return pd.DataFrame(data)


def test_clv_kpis_calculation(dummy_clv_df, dummy_master_df):
    """Verifies all 8 CLV KPI calculations."""
    engine = CLVKPIEngine()
    kpis = engine.compute_kpis(dummy_clv_df, master_df=dummy_master_df)

    assert len(kpis) == 8
    assert "total_predicted_clv" in kpis
    assert "avg_customer_clv" in kpis
    assert "highest_value_customer" in kpis
    assert "high_value_customers" in kpis
    assert "platinum_customers" in kpis
    assert "expected_revenue_12m" in kpis
    assert "avg_revenue_per_customer" in kpis
    assert "revenue_growth_potential" in kpis

    tot = kpis["total_predicted_clv"]
    assert tot["raw_value"] > 0
    assert "change_pct" in tot
    assert "trend_arrow" in tot


def test_value_classifier(dummy_clv_df):
    """Verifies 5-tier customer value classification and matrix summarization."""
    classifier = ValueClassifier()

    assert classifier.classify_customer_tier(3000.0) == "Platinum"
    assert classifier.classify_customer_tier(1500.0) == "Gold"
    assert classifier.classify_customer_tier(800.0) == "Silver"
    assert classifier.classify_customer_tier(400.0) == "Bronze"
    assert classifier.classify_customer_tier(150.0) == "Standard"

    matrix_df = classifier.get_value_tier_matrix(dummy_clv_df)
    assert not matrix_df.empty
    assert "Value_Tier" in matrix_df.columns
    assert "Customer_Count" in matrix_df.columns
    assert "Total_Revenue" in matrix_df.columns


def test_opportunity_intelligence(dummy_clv_df):
    """Verifies 6 strategic opportunity intelligence recommendations."""
    engine = OpportunityIntelligenceEngine()

    opps = engine.generate_opportunity_recommendations(dummy_clv_df)
    assert isinstance(opps, list)
    assert len(opps) == 6
    assert "priority" in opps[0]
    assert "estimated_revenue_impact" in opps[0]
    assert "confidence_score" in opps[0]


def test_clv_explainability_engine(dummy_clv_df):
    """Verifies Global SHAP Feature Importance and Local Diagnostic Explanation for CLV regression."""
    engine = CLVExplainabilityEngine()

    global_shap = engine.get_global_clv_feature_importance(dummy_clv_df)
    assert not global_shap.empty
    assert "Feature" in global_shap.columns
    assert "Importance_Weight" in global_shap.columns

    cid = dummy_clv_df["customer_unique_id"].iloc[0]
    explanation = engine.explain_customer_clv(cid, dummy_clv_df)
    assert explanation["customer_id"] == cid
    assert "plain_english_summary" in explanation
    assert "top_positive_drivers" in explanation


def test_revenue_forecast_engine(dummy_master_df, dummy_clv_df):
    """Verifies Monthly, Quarterly, and Annual revenue forecasting."""
    engine = RevenueForecastEngine()

    monthly = engine.get_revenue_forecast(dummy_master_df, dummy_clv_df, period_type="Monthly")
    assert "forecast_data" in monthly
    assert not monthly["forecast_data"].empty

    quarterly = engine.get_revenue_forecast(dummy_master_df, dummy_clv_df, period_type="Quarterly")
    assert not quarterly["forecast_data"].empty


def test_clv_search_engine(dummy_clv_df, dummy_master_df):
    """Verifies search by Customer ID, Customer Name, or Segment."""
    service = CLVService()

    res = service.search_clv_profile("CUST_001")
    assert res["has_match"]
    assert res["match_type"] == "Customer Account"
    assert "profile" in res
    assert "explanation" in res


def test_clv_export(dummy_clv_df):
    """Verifies CSV, Excel, and PDF exports for CLV Dashboard."""
    service = CLVService()

    csv_bytes, fn, mime = service.generate_export_file("csv", dummy_clv_df)
    assert len(csv_bytes) > 0
    assert mime == "text/csv"

    excel_bytes, fn, mime = service.generate_export_file("excel", dummy_clv_df)
    assert len(excel_bytes) > 0

    pdf_bytes, fn, mime = service.generate_export_file("pdf", dummy_clv_df)
    assert len(pdf_bytes) > 0


def test_empty_clv_dataset_handling():
    """Verifies system handles empty dataframes gracefully without crashing."""
    service = CLVService()
    empty_df = pd.DataFrame()

    engine = CLVKPIEngine()
    empty_kpis = engine.compute_kpis(empty_df)
    assert len(empty_kpis) == 8
    assert empty_kpis["total_predicted_clv"]["value"] == "$0.00"

    search_res = service.search_clv_profile("NON_EXISTENT_CUST_999")
    assert not search_res["has_match"]
