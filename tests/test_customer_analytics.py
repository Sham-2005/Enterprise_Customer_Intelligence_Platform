"""
Unit & Integration Test Suite for Phase 12 - Customer Analytics Module.
Tests 10 Customer KPIs, Demographics Engine, Behavior Engine, Loyalty Engine,
Pareto (80/20) Analysis, Activity Engine, Search Engine, Filters, Exports,
and Graceful Missing Dataset Handling.
"""

import pytest
import pandas as pd
import numpy as np
from backend.customer_analytics.customer_kpi_engine import CustomerKPIEngine
from backend.customer_analytics.demographics_engine import CustomerDemographicsEngine
from backend.customer_analytics.behavior_engine import CustomerBehaviorEngine
from backend.customer_analytics.loyalty_engine import CustomerLoyaltyEngine
from backend.customer_analytics.revenue_contribution_engine import RevenueContributionEngine
from backend.customer_analytics.activity_engine import CustomerActivityEngine
from backend.services.customer_analytics_service import CustomerAnalyticsService

@pytest.fixture
def dummy_master():
    """Generates synthetic master transaction dataset."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = {
        "order_id": [f"ORD_{i:04d}" for i in range(100)],
        "customer_unique_id": [f"CUST_{i%20:03d}" for i in range(100)],
        "customer_id": [f"CUST_ID_{i%20:03d}" for i in range(100)],
        "price": np.random.uniform(20, 400, size=100),
        "order_purchase_timestamp": dates,
        "customer_state": np.random.choice(["SP", "RJ", "MG", "RS"], size=100),
        "customer_city": np.random.choice(["sao paulo", "rio de janeiro", "belo horizonte"], size=100),
        "product_category_name_english": np.random.choice(["bed_bath_table", "health_beauty", "sports_leisure"], size=100),
        "payment_type": np.random.choice(["credit_card", "boleto"], size=100),
        "avg_review_score": np.random.choice([1.0, 3.0, 4.0, 5.0], size=100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_feature_store():
    """Generates synthetic feature store dataset."""
    data = {
        "customer_unique_id": [f"CUST_{i:03d}" for i in range(20)],
        "total_orders": np.random.randint(1, 8, size=20),
        "total_spending": np.random.uniform(100, 3000, size=20),
        "avg_order_value": np.random.uniform(50, 400, size=20),
        "recency_days": np.random.randint(5, 200, size=20),
        "churn_label": np.random.choice([0, 1], size=20),
        "rfm_segment": np.random.choice(["Champions", "Loyal Customers", "At Risk"], size=20),
        "historical_clv": np.random.uniform(100, 3000, size=20),
        "is_repeat_customer": np.random.choice([True, False], size=20)
    }
    return pd.DataFrame(data)


def test_customer_kpis_calculation(dummy_master, dummy_feature_store):
    """Verifies all 10 Customer KPI calculations."""
    engine = CustomerKPIEngine()
    kpis = engine.compute_kpis(dummy_master, dummy_feature_store)

    assert len(kpis) == 10
    assert "total_customers" in kpis
    assert "active_customers" in kpis
    assert "returning_customers" in kpis
    assert "new_customers" in kpis
    assert "repeat_purchase_rate" in kpis
    assert "avg_customer_clv" in kpis
    assert "avg_customer_rating" in kpis
    assert "customer_retention_rate" in kpis
    assert "avg_purchase_frequency" in kpis
    assert "avg_basket_size" in kpis

    tot = kpis["total_customers"]
    assert tot["raw_value"] > 0
    assert "change_pct" in tot
    assert "trend_arrow" in tot
    assert "last_updated" in tot


def test_demographics_engine(dummy_master, dummy_feature_store):
    """Verifies State & City demographics engine outputs."""
    engine = CustomerDemographicsEngine()

    state_df = engine.get_state_distribution(dummy_master, dummy_feature_store)
    assert not state_df.empty
    assert "State" in state_df.columns
    assert "Lat" in state_df.columns
    assert "Lon" in state_df.columns

    city_df = engine.get_city_distribution(dummy_master, top_n=10)
    assert not city_df.empty
    assert "City" in city_df.columns


def test_behavior_engine(dummy_master, dummy_feature_store):
    """Verifies purchasing behavior engine outputs."""
    engine = CustomerBehaviorEngine()

    freq_df = engine.get_purchase_frequency_distribution(dummy_master, dummy_feature_store)
    assert not freq_df.empty
    assert "Frequency_Bucket" in freq_df.columns

    div_df = engine.get_product_diversity_distribution(dummy_master)
    assert not div_df.empty

    pmt_df = engine.get_preferred_payment_methods(dummy_master)
    assert not pmt_df.empty


def test_loyalty_engine(dummy_master, dummy_feature_store):
    """Verifies loyalty tier categorization and histogram."""
    engine = CustomerLoyaltyEngine()

    tiers_df = engine.categorize_loyalty_tiers(dummy_feature_store, dummy_master)
    assert not tiers_df.empty
    assert "Loyalty_Tier" in tiers_df.columns

    hist_df = engine.get_loyalty_score_histogram(dummy_feature_store)
    assert not hist_df.empty


def test_revenue_contribution_and_pareto(dummy_master, dummy_feature_store):
    """Verifies Pareto 80/20 analysis and top customers ranking."""
    engine = RevenueContributionEngine()

    top_20 = engine.get_top_customers_by_revenue(dummy_master, dummy_feature_store, top_n=20)
    assert not top_20.empty
    assert len(top_20) <= 20

    pareto_res = engine.get_pareto_analysis(dummy_master, dummy_feature_store)
    assert "pareto_df" in pareto_res
    assert "top_20_rev_pct" in pareto_res
    assert pareto_res["top_20_rev_pct"] > 0.0


def test_activity_engine(dummy_master, dummy_feature_store):
    """Verifies recency distribution, active roster, and dormant customer extraction."""
    engine = CustomerActivityEngine()

    rec_df = engine.get_recency_distribution(dummy_feature_store, dummy_master)
    assert not rec_df.empty

    active_df = engine.get_recently_active_customers(dummy_master, dummy_feature_store, top_n=10)
    assert not active_df.empty

    dormant_df = engine.get_dormant_customers(dummy_master, dummy_feature_store, top_n=10)
    assert isinstance(dormant_df, pd.DataFrame)


def test_customer_search_engine(dummy_master, dummy_feature_store):
    """Verifies customer search by Customer ID, City, or State."""
    service = CustomerAnalyticsService()

    # Search Customer ID
    res_cust = service.search_customer_profile("CUST_001")
    assert res_cust["has_match"]
    assert res_cust["match_type"] == "Customer ID"
    assert "Customer ID" in res_cust["profile"]

    # Search City
    res_city = service.search_customer_profile("sao paulo")
    assert res_city["has_match"]
    assert res_city["match_type"] == "City"


def test_customer_analytics_export(dummy_master):
    """Verifies CSV, Excel, and PDF exports for Customer Analytics."""
    service = CustomerAnalyticsService()

    csv_bytes, fn, mime = service.generate_export_file("csv", dummy_master)
    assert len(csv_bytes) > 0
    assert mime == "text/csv"

    excel_bytes, fn, mime = service.generate_export_file("excel", dummy_master)
    assert len(excel_bytes) > 0

    pdf_bytes, fn, mime = service.generate_export_file("pdf", dummy_master)
    assert len(pdf_bytes) > 0


def test_empty_dataset_handling():
    """Verifies system handles empty dataframes gracefully without crashing."""
    service = CustomerAnalyticsService()
    empty_df = pd.DataFrame()

    engine = CustomerKPIEngine()
    empty_kpis = engine.compute_kpis(empty_df)
    assert len(empty_kpis) == 10
    assert empty_kpis["total_customers"]["value"] == "0"

    search_res = service.search_customer_profile("NON_EXISTENT_99999")
    assert not search_res["has_match"]
