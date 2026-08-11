"""
Unit & Integration Test Suite for Phase 13 - Customer Segmentation & RFM Intelligence Module.
Tests 8 Segmentation KPIs, Cluster Explorer, PCA 2D/3D Plot Generation, RFM Heatmap Engine,
Personas Manager, Marketing Intelligence Recommendations, Search Engine, Filters, Exports,
and Graceful Missing Dataset Handling.
"""

import pytest
import pandas as pd
import numpy as np
from backend.segmentation.segmentation_kpi_engine import SegmentationKPIEngine
from backend.segmentation.cluster_explorer_engine import ClusterExplorerEngine
from backend.segmentation.rfm_dashboard_engine import RFMDashboardEngine
from backend.segmentation.persona_manager import PersonaManager
from backend.segmentation.marketing_intelligence import MarketingIntelligenceEngine
from backend.services.segmentation_service import SegmentationService

@pytest.fixture
def dummy_feature_store():
    """Generates synthetic feature store dataset."""
    data = {
        "customer_unique_id": [f"CUST_{i:03d}" for i in range(30)],
        "total_spending": np.random.uniform(100, 4000, size=30),
        "total_orders": np.random.randint(1, 10, size=30),
        "avg_order_value": np.random.uniform(50, 400, size=30),
        "recency_days": np.random.randint(1, 180, size=30),
        "historical_clv": np.random.uniform(200, 5000, size=30),
        "predicted_clv": np.random.uniform(200, 5000, size=30),
        "avg_review_score_given": np.random.uniform(2.0, 5.0, size=30),
        "loyalty_score": np.random.uniform(10, 100, size=30),
        "churn_label": np.random.choice([0, 1], size=30),
        "cluster_name": np.random.choice(["Champions", "Loyal Customers", "At Risk", "Hibernating"], size=30),
        "rfm_segment": np.random.choice(["Champions", "Loyal Customers", "At Risk", "Hibernating"], size=30),
        "r_score": np.random.choice([1, 2, 3, 4, 5], size=30),
        "f_score": np.random.choice([1, 2, 3, 4, 5], size=30),
        "m_score": np.random.choice([1, 2, 3, 4, 5], size=30)
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_master():
    """Generates synthetic master dataset."""
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


def test_segmentation_kpis_calculation(dummy_feature_store, dummy_master):
    """Verifies all 8 Segmentation KPI calculations."""
    engine = SegmentationKPIEngine()
    kpis = engine.compute_kpis(dummy_feature_store, master_df=dummy_master)

    assert len(kpis) == 8
    assert "total_segments" in kpis
    assert "total_customers_clustered" in kpis
    assert "vip_customers" in kpis
    assert "loyal_customers" in kpis
    assert "at_risk_customers" in kpis
    assert "avg_cluster_revenue" in kpis
    assert "avg_rfm_score" in kpis
    assert "largest_customer_segment" in kpis

    tot = kpis["total_customers_clustered"]
    assert tot["raw_value"] > 0
    assert "change_pct" in tot
    assert "trend_arrow" in tot


def test_cluster_explorer_engine(dummy_feature_store, dummy_master):
    """Verifies Cluster Overview, Details, PCA 2D/3D projections, and Cluster Comparison Matrix."""
    engine = ClusterExplorerEngine()

    overview = engine.get_cluster_overview(dummy_feature_store, dummy_master)
    assert "distribution" in overview
    assert "revenue_treemap" in overview
    assert "averages" in overview

    details = engine.get_cluster_details("Champions", dummy_feature_store, dummy_master)
    assert details["cluster_name"] == "Champions"
    assert "number_of_customers" in details

    pca_df = engine.get_pca_visualization_data(dummy_feature_store, sample_size=100)
    assert not pca_df.empty
    assert "PC1" in pca_df.columns
    assert "PC2" in pca_df.columns
    assert "PC3" in pca_df.columns

    comp_df = engine.get_cluster_comparison_matrix(dummy_feature_store)
    assert not comp_df.empty
    assert "Cluster_Name" in comp_df.columns


def test_rfm_dashboard_engine(dummy_feature_store):
    """Verifies RFM Quintiles, RFM Heatmap Matrix, and RFM Segment Breakdown."""
    engine = RFMDashboardEngine()

    quintiles = engine.get_rfm_quintiles_distribution(dummy_feature_store)
    assert "recency" in quintiles
    assert "frequency" in quintiles
    assert "monetary" in quintiles

    heatmap = engine.get_rfm_heatmap_matrix(dummy_feature_store)
    assert not heatmap.empty

    segments_df = engine.get_rfm_segment_distribution(dummy_feature_store)
    assert not segments_df.empty
    assert "RFM_Segment" in segments_df.columns


def test_persona_manager(dummy_feature_store):
    """Verifies Persona definitions and blueprint outputs."""
    manager = PersonaManager()

    personas = manager.get_personas_summary(dummy_feature_store)
    assert isinstance(personas, list)
    assert len(personas) > 0
    assert "persona_title" in personas[0]
    assert "marketing_recommendation" in personas[0]


def test_marketing_intelligence(dummy_feature_store, dummy_master):
    """Verifies 6 automated marketing intelligence recommendation cards."""
    engine = MarketingIntelligenceEngine()

    recommendations = engine.generate_recommendations(dummy_feature_store, dummy_master)
    assert isinstance(recommendations, list)
    assert len(recommendations) == 6
    assert "title" in recommendations[0]
    assert "action_plan" in recommendations[0]


def test_segmentation_search_engine(dummy_feature_store, dummy_master):
    """Verifies search by Customer ID, Cluster Name, or Persona."""
    service = SegmentationService()

    # Search Customer ID
    res_cust = service.search_segmentation_profile("CUST_001")
    assert res_cust["has_match"]
    assert res_cust["match_type"] == "Customer ID"

    # Search Cluster Name
    res_clust = service.search_segmentation_profile("Champions")
    assert res_clust["has_match"]
    assert res_clust["match_type"] == "Cluster / Persona"


def test_segmentation_export(dummy_feature_store):
    """Verifies CSV, Excel, and PDF exports for Segmentation."""
    service = SegmentationService()

    csv_bytes, fn, mime = service.generate_export_file("csv", dummy_feature_store)
    assert len(csv_bytes) > 0
    assert mime == "text/csv"

    excel_bytes, fn, mime = service.generate_export_file("excel", dummy_feature_store)
    assert len(excel_bytes) > 0

    pdf_bytes, fn, mime = service.generate_export_file("pdf", dummy_feature_store)
    assert len(pdf_bytes) > 0


def test_empty_segmentation_dataset_handling():
    """Verifies system handles empty dataframes gracefully without crashing."""
    service = SegmentationService()
    empty_df = pd.DataFrame()

    engine = SegmentationKPIEngine()
    empty_kpis = engine.compute_kpis(empty_df)
    assert len(empty_kpis) == 8
    assert empty_kpis["total_customers_clustered"]["value"] == "0"

    search_res = service.search_segmentation_profile("NON_EXISTENT_CLUSTER_999")
    assert not search_res["has_match"]
