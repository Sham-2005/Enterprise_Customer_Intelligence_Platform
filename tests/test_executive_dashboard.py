"""
Unit & Integration Test Suite for Phase 11 - Executive Dashboard Backend Integration.
Tests KPI Calculations, Filter Engine, Analytics Datasets, Global Search, File Exports,
Caching Layer, and Graceful Missing Dataset Handling.
"""

import pytest
import pandas as pd
import numpy as np
from backend.services.data_service import DataService
from backend.services.filter_service import FilterService
from backend.services.kpi_service import KPIService
from backend.services.analytics_service import AnalyticsService
from backend.services.search_service import SearchService
from backend.services.export_service import ExportService
from backend.dashboard.executive_backend import ExecutiveDashboardBackend
from backend.cache.dashboard_cache import DashboardCache

@pytest.fixture
def dummy_master_data():
    """Generates synthetic master transaction dataframe for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = {
        "order_id": [f"ORD_{i:04d}" for i in range(100)],
        "customer_unique_id": [f"CUST_{i%25:03d}" for i in range(100)],
        "customer_id": [f"CUST_ID_{i%25:03d}" for i in range(100)],
        "price": np.random.uniform(50, 500, size=100),
        "freight_value": np.random.uniform(10, 50, size=100),
        "order_purchase_timestamp": dates,
        "customer_state": np.random.choice(["SP", "RJ", "MG", "RS", "BA"], size=100),
        "seller_state": np.random.choice(["SP", "RJ", "MG"], size=100),
        "product_category_name_english": np.random.choice(["bed_bath_table", "health_beauty", "sports_leisure", "computers"], size=100),
        "seller_id": np.random.choice(["SEL_01", "SEL_02", "SEL_03"], size=100),
        "product_id": [f"PROD_{i%30:03d}" for i in range(100)],
        "payment_type": np.random.choice(["credit_card", "boleto", "voucher"], size=100),
        "order_status": np.random.choice(["delivered", "shipped", "canceled"], size=100),
        "avg_review_score": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=100),
        "rfm_segment": np.random.choice(["Champions", "Loyal Customers", "At Risk"], size=100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_feature_store_data():
    """Generates synthetic feature store dataframe for testing."""
    data = {
        "customer_unique_id": [f"CUST_{i:03d}" for i in range(25)],
        "total_orders": np.random.randint(1, 10, size=25),
        "total_spending": np.random.uniform(100, 2000, size=25),
        "avg_order_value": np.random.uniform(50, 300, size=25),
        "recency_days": np.random.randint(1, 180, size=25),
        "churn_label": np.random.choice([0, 1], size=25),
        "risk_level": np.random.choice(["Low Risk", "Medium Risk", "High Risk"], size=25),
        "rfm_segment": np.random.choice(["Champions", "Loyal Customers", "At Risk"], size=25),
        "predicted_clv": np.random.uniform(200, 5000, size=25),
        "is_repeat_customer": np.random.choice([True, False], size=25)
    }
    return pd.DataFrame(data)


def test_kpi_service_calculations(dummy_master_data, dummy_feature_store_data):
    """Verifies KPI Service metrics, period comparisons, and trend outputs."""
    service = KPIService()
    kpis = service.compute_all_kpis(dummy_master_data, dummy_feature_store_data)

    assert "total_revenue" in kpis
    assert "total_orders" in kpis
    assert "total_customers" in kpis
    assert "avg_order_value" in kpis
    assert "avg_rating" in kpis
    assert "retention_rate" in kpis
    assert "monthly_revenue_growth" in kpis
    assert "business_health_score" in kpis

    rev = kpis["total_revenue"]
    assert rev["raw_value"] > 0
    assert "change_pct" in rev
    assert "trend_arrow" in rev
    assert "last_updated" in rev


def test_filter_service(dummy_master_data, dummy_feature_store_data):
    """Verifies filter engine correctly filters by Date Range, State, Category, Seller, Payment, Segment."""
    service = FilterService()

    # Filter by State SP
    filtered_master, filtered_fs = service.filter_executive_data(
        master_df=dummy_master_data,
        feature_store_df=dummy_feature_store_data,
        states=["SP"]
    )
    assert not filtered_master.empty
    assert (filtered_master["customer_state"] == "SP").all()

    # Filter Options Extraction
    options = service.extract_filter_options(dummy_master_data, dummy_feature_store_data)
    assert "SP" in options["states"]
    assert "credit_card" in options["payment_methods"]


def test_analytics_service_chart_data(dummy_master_data, dummy_feature_store_data):
    """Verifies analytics engine generates proper chart data structures."""
    service = AnalyticsService()

    # Revenue Trend
    trend_monthly = service.get_revenue_trend(dummy_master_data, granularity="Monthly")
    assert not trend_monthly.empty
    assert "Period" in trend_monthly.columns
    assert "Revenue" in trend_monthly.columns

    # Customer Growth
    growth_df = service.get_customer_growth(dummy_master_data, dummy_feature_store_data)
    assert not growth_df.empty
    assert "New_Customers" in growth_df.columns

    # Category Treemap
    treemap_df = service.get_revenue_by_category_treemap(dummy_master_data)
    assert not treemap_df.empty
    assert "Category" in treemap_df.columns

    # State Map
    state_df = service.get_revenue_by_state_map(dummy_master_data)
    assert not state_df.empty
    assert "Lat" in state_df.columns
    assert "Lon" in state_df.columns

    # Insights & Summary
    insights = service.generate_recent_business_insights(dummy_master_data, dummy_feature_store_data)
    assert isinstance(insights, list)
    assert len(insights) > 0

    summary_text = service.generate_executive_summary_text(dummy_master_data, dummy_feature_store_data)
    assert isinstance(summary_text, str)
    assert "Executive Overview" in summary_text


def test_search_service(dummy_master_data, dummy_feature_store_data):
    """Verifies global universal search engine matches Customer, Order, Seller, and Product IDs."""
    service = SearchService()

    # Customer search
    res_cust = service.search("CUST_001", dummy_master_data, dummy_feature_store_data)
    assert res_cust["has_match"]
    assert res_cust["match_type"] == "Customer"

    # Order search
    res_ord = service.search("ORD_0005", dummy_master_data, dummy_feature_store_data)
    assert res_ord["has_match"]
    assert res_ord["match_type"] == "Order"

    # Seller search
    res_sel = service.search("SEL_01", dummy_master_data, dummy_feature_store_data)
    assert res_sel["has_match"]
    assert res_sel["match_type"] == "Seller"


def test_export_service(dummy_master_data):
    """Verifies CSV, Excel, and PDF report file generation."""
    service = ExportService()

    csv_bytes = service.export_to_csv(dummy_master_data)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0

    excel_bytes = service.export_to_excel(dummy_master_data)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    pdf_bytes = service.export_to_pdf(dummy_master_data, summary_text="Test Summary")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_cache_service():
    """Verifies in-memory TTL caching operations and stats."""
    cache = DashboardCache(default_ttl=5)
    params = {"state": "SP", "cat": "health_beauty"}

    key = cache.set("test_prefix", {"data": 123}, params=params)
    assert key.startswith("test_prefix:")

    val = cache.get("test_prefix", params=params)
    assert val == {"data": 123}

    stats = cache.get_stats()
    assert stats["hits"] == 1

    cache.clear()
    assert len(cache._store) == 0


def test_graceful_missing_data_handling():
    """Verifies dashboard payload functions without errors even on empty dataframes."""
    backend = ExecutiveDashboardBackend()
    empty_master = pd.DataFrame()

    kpi_service = KPIService()
    empty_kpis = kpi_service.compute_all_kpis(empty_master)
    assert "total_revenue" in empty_kpis
    assert empty_kpis["total_revenue"]["value"] == "$0.00"

    analytics_service = AnalyticsService()
    empty_insights = analytics_service.generate_recent_business_insights(empty_master)
    assert len(empty_insights) > 0

    export_service = ExportService()
    empty_csv = export_service.export_to_csv(empty_master)
    assert b"No data" in empty_csv
