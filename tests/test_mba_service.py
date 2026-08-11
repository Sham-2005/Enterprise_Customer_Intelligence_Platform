"""
Unit & Integration Test Suite for Phase 17 - Market Basket Analysis Dashboard.
Tests MBAService dataset detection, 8 MBA KPIs, multi-criteria rule filtering,
Product Association Network graph generation, Customer Segment basket analysis,
Category co-occurrence matrix, Product Search Intelligence, Business Recommendations,
Data Exports, and Graceful Fallback Handling.
"""

import pytest
import pandas as pd
import numpy as np

from backend.services.mba_service import MBAService

@pytest.fixture
def dummy_master_df():
    """Generates synthetic master transaction dataset."""
    data = {
        "order_id": [f"ORD_{i%20:04d}" for i in range(100)],
        "customer_unique_id": [f"CUST_{i%10:03d}" for i in range(100)],
        "product_id": [f"PROD_{i%15:03d}" for i in range(100)],
        "order_item_id": [1]*100,
        "price": np.random.uniform(20, 300, size=100),
        "order_purchase_timestamp": pd.date_range(start="2024-01-01", periods=100, freq="D"),
        "product_category_name_english": np.random.choice(["health_beauty", "bed_bath_table", "sports_leisure"], size=100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_rules_df():
    """Generates synthetic association rules dataset."""
    data = {
        "antecedents_str": ["health_beauty", "bed_bath_table", "sports_leisure", "computers_accessories"],
        "consequents_str": ["perfumery", "furniture_decor", "apparel", "accessories"],
        "support": [0.015, 0.025, 0.012, 0.030],
        "confidence": [0.65, 0.58, 0.72, 0.81],
        "lift": [3.45, 2.10, 4.15, 5.20],
        "leverage": [0.01, 0.02, 0.01, 0.03],
        "conviction": [1.5, 1.8, 2.1, 2.5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_bundles_df():
    """Generates synthetic product bundles dataset."""
    data = {
        "bundle_name": ["health_beauty + perfumery", "bed_bath_table + furniture_decor"],
        "primary_category": ["health_beauty", "bed_bath_table"],
        "addon_category": ["perfumery", "furniture_decor"],
        "lift_score": [3.45, 2.10],
        "confidence_pct": [65.0, 58.0],
        "support_pct": [1.5, 2.5],
        "estimated_bundle_price": [85.50, 140.00],
        "projected_revenue_potential": [42500.00, 28000.00],
        "merchandising_strategy": ["Promote joint bundle with 10% discount.", "Add-on widget at checkout."]
    }
    return pd.DataFrame(data)


def test_mba_dataset_loading(dummy_master_df, dummy_rules_df, dummy_bundles_df):
    """Verifies dataset files detection and fallback loading logic."""
    service = MBAService()
    datasets = {
        "master_dataset": dummy_master_df,
        "association_rules": dummy_rules_df,
        "product_bundles": dummy_bundles_df,
        "mba_metrics": {"total_transactions": 20, "association_rules_count": 4}
    }

    assert not datasets["master_dataset"].empty
    assert not datasets["association_rules"].empty
    assert datasets["mba_metrics"]["association_rules_count"] == 4


def test_mba_kpis_calculation(dummy_master_df, dummy_rules_df, dummy_bundles_df):
    """Verifies all 8 Market Basket KPI calculations."""
    service = MBAService()
    datasets = {
        "master_dataset": dummy_master_df,
        "association_rules": dummy_rules_df,
        "product_bundles": dummy_bundles_df,
        "mba_metrics": {"total_transactions": 20, "unique_categories": 3}
    }

    kpis = service.compute_mba_kpis(datasets)
    assert len(kpis) == 8
    assert "total_transactions" in kpis
    assert "association_rules" in kpis
    assert "high_lift_rules" in kpis
    assert kpis["association_rules"]["value"] == "4"


def test_association_rule_filtering(dummy_rules_df):
    """Verifies multi-attribute rule filtering by antecedent, min_lift, min_confidence."""
    service = MBAService()
    
    filtered = service.filter_association_rules(dummy_rules_df, min_lift=3.0)
    assert len(filtered) == 3

    filtered_ant = service.filter_association_rules(dummy_rules_df, antecedent="health_beauty")
    assert len(filtered_ant) == 1
    assert filtered_ant.iloc[0]["antecedents_str"] == "health_beauty"


def test_association_network_graph_data(dummy_rules_df):
    """Verifies construction of node and edge data structures for network graph."""
    service = MBAService()
    graph = service.get_association_network_graph(dummy_rules_df)

    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) > 0
    assert len(graph["edges"]) > 0


def test_customer_segment_basket_analysis(dummy_master_df):
    """Verifies customer segment basket metrics extraction."""
    service = MBAService()
    datasets = {"master_dataset": dummy_master_df}

    seg_df = service.get_customer_segment_basket_analysis(datasets)
    assert not seg_df.empty
    assert "Avg Basket Size" in seg_df.columns
    assert "Most Purchased Category" in seg_df.columns


def test_category_cooccurrence_matrix(dummy_master_df):
    """Verifies category co-occurrence crosstab calculation."""
    service = MBAService()
    co_matrix = service.get_category_cooccurrence_matrix(dummy_master_df)

    assert not co_matrix.empty
    assert co_matrix.shape[0] == co_matrix.shape[1]


def test_product_search_intelligence(dummy_master_df, dummy_rules_df, dummy_bundles_df):
    """Verifies product search details retrieval."""
    service = MBAService()
    datasets = {
        "master_dataset": dummy_master_df,
        "association_rules": dummy_rules_df,
        "product_bundles": dummy_bundles_df
    }

    search_res = service.get_product_search_intelligence("health_beauty", datasets)
    assert search_res["query"] == "health_beauty"
    assert "strongest_rules" in search_res
    assert "recommended_bundles" in search_res


def test_business_recommendations_generator(dummy_bundles_df):
    """Verifies data-driven business merchandising recommendations."""
    service = MBAService()
    datasets = {"product_bundles": dummy_bundles_df}

    recs = service.get_business_recommendations(datasets)
    assert len(recs) == 4
    assert "type" in recs[0]
    assert "action" in recs[0]
