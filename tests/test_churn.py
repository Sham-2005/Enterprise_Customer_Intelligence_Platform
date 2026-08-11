"""
Unit & Integration Test Suite for Phase 14 - AI Customer Churn Prediction & Risk Intelligence Dashboard.
Tests 8 Churn KPIs, 5-Tier Risk Stratification, SHAP Explainable AI (XAI) Engine,
Personalized Retention Campaigns, Customer Timelines, Batch Predictor, Search Engine,
Filters, Exports, and Graceful Error Handling.
"""

import io
import pytest
import pandas as pd
import numpy as np
from backend.churn.churn_kpi_engine import ChurnKPIEngine
from backend.churn.risk_classifier import RiskClassifier
from backend.churn.retention_intelligence import RetentionIntelligenceEngine
from backend.churn.explainability_engine import ExplainabilityEngine
from backend.churn.customer_timeline_engine import CustomerTimelineEngine
from backend.churn.batch_predictor import BatchPredictor
from backend.services.churn_service import ChurnService

@pytest.fixture
def dummy_churn_df():
    """Generates synthetic customer churn prediction dataset."""
    data = {
        "customer_unique_id": [f"CUST_{i:03d}" for i in range(30)],
        "churn_probability": np.random.uniform(0.05, 0.95, size=30),
        "risk_level": np.random.choice(["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"], size=30),
        "total_spending": np.random.uniform(100, 5000, size=30),
        "total_orders": np.random.randint(1, 10, size=30),
        "avg_order_value": np.random.uniform(50, 400, size=30),
        "recency_days": np.random.randint(1, 180, size=30),
        "historical_clv": np.random.uniform(200, 6000, size=30),
        "predicted_clv": np.random.uniform(200, 6000, size=30),
        "avg_review_score_given": np.random.uniform(1.0, 5.0, size=30),
        "rfm_segment": np.random.choice(["Champions", "Loyal Customers", "At Risk", "Hibernating"], size=30)
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


def test_churn_kpis_calculation(dummy_churn_df, dummy_master_df):
    """Verifies all 8 Churn KPI calculations."""
    engine = ChurnKPIEngine()
    kpis = engine.compute_kpis(dummy_churn_df, master_df=dummy_master_df)

    assert len(kpis) == 8
    assert "total_customers" in kpis
    assert "high_risk_customers" in kpis
    assert "critical_risk_customers" in kpis
    assert "avg_churn_probability" in kpis
    assert "predicted_churn_rate" in kpis
    assert "retention_success_estimate" in kpis
    assert "avg_customer_clv" in kpis
    assert "estimated_revenue_at_risk" in kpis

    tot = kpis["total_customers"]
    assert tot["raw_value"] > 0
    assert "change_pct" in tot
    assert "trend_arrow" in tot


def test_risk_classifier_stratification(dummy_churn_df):
    """Verifies 5-tier risk classification and distribution table."""
    classifier = RiskClassifier()

    assert classifier.stratify_risk_level(0.15) == "Very Low Risk"
    assert classifier.stratify_risk_level(0.35) == "Low Risk"
    assert classifier.stratify_risk_level(0.55) == "Medium Risk"
    assert classifier.stratify_risk_level(0.75) == "High Risk"
    assert classifier.stratify_risk_level(0.95) == "Critical Risk"

    risk_df = classifier.get_risk_distribution(dummy_churn_df)
    assert not risk_df.empty
    assert "Risk_Tier" in risk_df.columns
    assert "Customer_Count" in risk_df.columns


def test_explainability_engine(dummy_churn_df):
    """Verifies SHAP Global Importance and Local Customer Narrative explanations."""
    engine = ExplainabilityEngine()

    global_shap = engine.get_global_feature_importance(dummy_churn_df)
    assert not global_shap.empty
    assert "Feature" in global_shap.columns
    assert "Importance_Weight" in global_shap.columns

    cid = dummy_churn_df["customer_unique_id"].iloc[0]
    explanation = engine.explain_customer(cid, dummy_churn_df)
    assert explanation["customer_id"] == cid
    assert "plain_english_explanation" in explanation
    assert "top_positive_risk_factors" in explanation


def test_retention_intelligence_recommendations(dummy_churn_df):
    """Verifies 8 personalized retention campaign recommendations."""
    engine = RetentionIntelligenceEngine()

    recs = engine.generate_retention_recommendations(dummy_churn_df)
    assert isinstance(recs, list)
    assert len(recs) == 8
    assert "priority" in recs[0]
    assert "estimated_impact" in recs[0]
    assert "expected_revenue_saved" in recs[0]
    assert "confidence_score" in recs[0]


def test_customer_timeline_engine(dummy_master_df, dummy_churn_df):
    """Verifies customer purchase timeline and risk trajectory curve."""
    engine = CustomerTimelineEngine()

    cid = dummy_churn_df["customer_unique_id"].iloc[0]
    timeline = engine.get_customer_timeline(cid, dummy_master_df, dummy_churn_df)

    assert timeline["customer_id"] == cid
    assert "timeline_events" in timeline
    assert "risk_trajectory" in timeline


def test_batch_predictor():
    """Verifies batch CSV prediction engine."""
    predictor = BatchPredictor()

    sample_csv = "total_spending,total_orders,recency_days\n500.0,3,45\n120.0,1,120\n"
    df = pd.read_csv(io.StringIO(sample_csv))

    scored_df, metrics = predictor.run_batch_prediction(df)
    assert not scored_df.empty
    assert "churn_probability" in scored_df.columns
    assert "risk_level" in scored_df.columns
    assert metrics["total_records"] == 2


def test_churn_search_engine(dummy_churn_df, dummy_master_df):
    """Verifies search by Customer ID."""
    service = ChurnService()

    res = service.search_churn_profile("CUST_001")
    assert res["has_match"]
    assert res["match_type"] == "Customer Account"
    assert "profile" in res
    assert "explanation" in res


def test_churn_export(dummy_churn_df):
    """Verifies CSV, Excel, and PDF exports for Churn Dashboard."""
    service = ChurnService()

    csv_bytes, fn, mime = service.generate_export_file("csv", dummy_churn_df)
    assert len(csv_bytes) > 0
    assert mime == "text/csv"

    excel_bytes, fn, mime = service.generate_export_file("excel", dummy_churn_df)
    assert len(excel_bytes) > 0

    pdf_bytes, fn, mime = service.generate_export_file("pdf", dummy_churn_df)
    assert len(pdf_bytes) > 0


def test_empty_churn_dataset_handling():
    """Verifies system handles empty dataframes gracefully without crashing."""
    service = ChurnService()
    empty_df = pd.DataFrame()

    engine = ChurnKPIEngine()
    empty_kpis = engine.compute_kpis(empty_df)
    assert len(empty_kpis) == 8
    assert empty_kpis["total_customers"]["value"] == "0"

    search_res = service.search_churn_profile("NON_EXISTENT_CUST_999")
    assert not search_res["has_match"]
