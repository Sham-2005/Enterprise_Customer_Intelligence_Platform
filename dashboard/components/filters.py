"""
Interactive Filter Panel & Universal Search UI Components for ECIP Executive, Customer Analytics, Segmentation, Churn & CLV Dashboard.
Provides sidebar filter controls for Date Ranges, States, Cities, Product Categories, Sellers,
Payment Methods, Customer Segments, Customer Tiers, Revenue Ranges, CLV Ranges, and Churn Risk Tiers.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st

def render_global_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders interactive sidebar filter controls for Executive Dashboard."""
    options = filter_options or {}

    st.sidebar.markdown("### 🎛️ Executive Filters")

    with st.sidebar.expander("📅 Date Range Filter", expanded=True):
        date_range = st.date_input("Select Period", [], key="exec_filter_date_range")

    with st.sidebar.expander("🗺️ Geography (State)", expanded=False):
        state_opts = options.get("states", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA"])
        selected_states = st.multiselect("Select States", options=state_opts, default=[], key="exec_filter_states")

    with st.sidebar.expander("🛍️ Product Category", expanded=False):
        cat_opts = options.get("categories", [])
        selected_cats = st.multiselect("Select Categories", options=cat_opts, default=[], key="exec_filter_cats")

    with st.sidebar.expander("🏪 Seller Lookup", expanded=False):
        seller_opts = options.get("sellers", [])
        selected_sellers = st.multiselect("Select Sellers", options=seller_opts[:100], default=[], key="exec_filter_sellers")

    with st.sidebar.expander("💳 Payment Method", expanded=False):
        pmt_opts = options.get("payment_methods", ["credit_card", "boleto", "voucher", "debit_card"])
        selected_pmts = st.multiselect("Select Payment Methods", options=pmt_opts, default=[], key="exec_filter_pmts")

    with st.sidebar.expander("🎯 Customer Segment", expanded=False):
        seg_opts = options.get("customer_segments", ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"])
        selected_segs = st.multiselect("Select Segments", options=seg_opts, default=[], key="exec_filter_segs")

    st.sidebar.markdown("---")

    return {
        "date_range": date_range,
        "states": selected_states,
        "categories": selected_cats,
        "sellers": selected_sellers,
        "payment_methods": selected_pmts,
        "customer_segments": selected_segs
    }


def render_customer_analytics_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders sidebar filter panel for Customer Analytics."""
    options = filter_options or {}

    st.sidebar.markdown("### 👥 Customer Analytics Filters")

    with st.sidebar.expander("📅 Date Range Filter", expanded=True):
        date_range = st.sidebar.date_input("Select Period", [], key="cust_filter_date_range")

    with st.sidebar.expander("🗺️ Geography (State & City)", expanded=False):
        state_opts = options.get("states", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA"])
        selected_states = st.multiselect("Customer State", options=state_opts, default=[], key="cust_filter_states")
        city_opts = options.get("cities", [])
        selected_cities = st.multiselect("Customer City", options=city_opts[:100], default=[], key="cust_filter_cities")

    with st.sidebar.expander("🎯 Customer Segment", expanded=False):
        seg_opts = options.get("customer_segments", ["VIP Customers", "High-Value Customers", "Loyal Customers", "Occasional Customers", "One-Time Buyers"])
        selected_segs = st.multiselect("Select Segments", options=seg_opts, default=[], key="cust_filter_segs")

    with st.sidebar.expander("💳 Payment Method", expanded=False):
        pmt_opts = options.get("payment_methods", ["credit_card", "boleto", "voucher", "debit_card"])
        selected_pmts = st.multiselect("Payment Method", options=pmt_opts, default=[], key="cust_filter_pmts")

    with st.sidebar.expander("🛍️ Product Category", expanded=False):
        cat_opts = options.get("categories", [])
        selected_cats = st.multiselect("Category Purchased", options=cat_opts, default=[], key="cust_filter_cats")

    with st.sidebar.expander("💰 Customer Spending Range ($)", expanded=False):
        rev_range = st.slider("Total Spending ($)", min_value=0, max_value=5000, value=(0, 5000), key="cust_filter_rev_slider")

    st.sidebar.markdown("---")

    return {
        "date_range": date_range,
        "states": selected_states,
        "cities": selected_cities,
        "customer_segments": selected_segs,
        "payment_methods": selected_pmts,
        "categories": selected_cats,
        "revenue_range": rev_range
    }


def render_segmentation_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders sidebar filter panel for Customer Segmentation & RFM Module."""
    options = filter_options or {}

    st.sidebar.markdown("### 🧩 Segmentation & RFM Filters")

    with st.sidebar.expander("🧩 Cluster & Persona", expanded=True):
        clust_opts = options.get("customer_segments", ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating", "Cluster 0", "Cluster 1", "Cluster 2"])
        selected_clusters = st.multiselect("Select Clusters", options=clust_opts, default=[], key="seg_filter_clusters")
        persona_opts = ["VIP Power Buyers", "Loyal Frequent Buyers", "Premium Customers", "New Customers", "Occasional Buyers", "Price Sensitive Customers", "At-Risk Customers", "Lost Customers"]
        selected_personas = st.multiselect("Select Personas", options=persona_opts, default=[], key="seg_filter_personas")

    with st.sidebar.expander("🗺️ State & Product Category", expanded=False):
        state_opts = options.get("states", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA"])
        selected_states = st.multiselect("Select States", options=state_opts, default=[], key="seg_filter_states")
        cat_opts = options.get("categories", [])
        selected_cats = st.multiselect("Select Categories", options=cat_opts, default=[], key="seg_filter_cats")

    with st.sidebar.expander("💰 Revenue & CLV Tiers ($)", expanded=False):
        rev_range = st.slider("Revenue Range ($)", min_value=0, max_value=5000, value=(0, 5000), key="seg_filter_rev_slider")
        clv_range = st.slider("CLV Range ($)", min_value=0, max_value=10000, value=(0, 10000), key="seg_filter_clv_slider")

    with st.sidebar.expander("⚠️ Churn Risk & Date Range", expanded=False):
        risk_opts = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
        selected_risks = st.multiselect("Churn Risk Tiers", options=risk_opts, default=[], key="seg_filter_risks")
        date_range = st.date_input("Date Range", [], key="seg_filter_date_range")

    st.sidebar.markdown("---")

    return {
        "clusters": selected_clusters,
        "personas": selected_personas,
        "states": selected_states,
        "categories": selected_cats,
        "revenue_range": rev_range,
        "clv_range": clv_range,
        "churn_risk": selected_risks,
        "date_range": date_range
    }


def render_churn_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders sidebar filter panel for Churn Prediction Dashboard."""
    options = filter_options or {}

    st.sidebar.markdown("### 🤖 Churn & Risk Filters")

    with st.sidebar.expander("⚠️ Risk Tier Filter", expanded=True):
        risk_opts = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
        selected_risks = st.multiselect("Select Risk Tiers", options=risk_opts, default=[], key="churn_filter_risks")

    with st.sidebar.expander("🎯 Segment & State", expanded=False):
        seg_opts = options.get("customer_segments", ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"])
        selected_segs = st.multiselect("Select Segments", options=seg_opts, default=[], key="churn_filter_segs")
        state_opts = options.get("states", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA"])
        selected_states = st.multiselect("Select States", options=state_opts, default=[], key="churn_filter_states")

    with st.sidebar.expander("🛍️ Category & Date Range", expanded=False):
        cat_opts = options.get("categories", [])
        selected_cats = st.multiselect("Select Categories", options=cat_opts, default=[], key="churn_filter_cats")
        date_range = st.date_input("Select Period", [], key="churn_filter_date_range")

    with st.sidebar.expander("💰 Revenue & CLV Tiers ($)", expanded=False):
        rev_range = st.slider("Total Revenue ($)", min_value=0, max_value=5000, value=(0, 5000), key="churn_filter_rev_slider")
        clv_range = st.slider("CLV Range ($)", min_value=0, max_value=10000, value=(0, 10000), key="churn_filter_clv_slider")

    st.sidebar.markdown("---")

    return {
        "risk_levels": selected_risks,
        "clusters": selected_segs,
        "states": selected_states,
        "categories": selected_cats,
        "date_range": date_range,
        "revenue_range": rev_range,
        "clv_range": clv_range
    }


def render_clv_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders sidebar filter panel for Customer Lifetime Value (CLV) & Revenue Intelligence Dashboard."""
    options = filter_options or {}

    st.sidebar.markdown("### 💎 CLV & Revenue Filters")

    # 1. Customer Value Tier Filter
    with st.sidebar.expander("🏆 Customer Value Tier", expanded=True):
        tier_opts = ["Platinum", "Gold", "Silver", "Bronze", "Standard"]
        selected_tiers = st.multiselect("Select Value Tiers", options=tier_opts, default=[], key="clv_filter_tiers")

    # 2. Forecast Period Selector
    with st.sidebar.expander("📈 Revenue Forecast Horizon", expanded=True):
        forecast_period = st.selectbox("Select Forecast Horizon", options=["Monthly", "Quarterly", "Annual"], index=0, key="clv_filter_forecast_period")

    # 3. Customer Segment & State
    with st.sidebar.expander("🎯 Segment & State", expanded=False):
        seg_opts = options.get("customer_segments", ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"])
        selected_segs = st.multiselect("Select Segments", options=seg_opts, default=[], key="clv_filter_segs")

        state_opts = options.get("states", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA"])
        selected_states = st.multiselect("Select States", options=state_opts, default=[], key="clv_filter_states")

    # 4. Product Category & Date Range
    with st.sidebar.expander("🛍️ Category & Date Range", expanded=False):
        cat_opts = options.get("categories", [])
        selected_cats = st.multiselect("Select Categories", options=cat_opts, default=[], key="clv_filter_cats")
        date_range = st.date_input("Date Range", [], key="clv_filter_date_range")

    # 5. Revenue & CLV Range Slider
    with st.sidebar.expander("💰 Revenue & CLV Tiers ($)", expanded=False):
        rev_range = st.slider("Revenue Range ($)", min_value=0, max_value=5000, value=(0, 5000), key="clv_filter_rev_slider")
        clv_range = st.slider("CLV Range ($)", min_value=0, max_value=10000, value=(0, 10000), key="clv_filter_clv_slider")

    st.sidebar.markdown("---")

    return {
        "tiers": selected_tiers,
        "forecast_period": forecast_period,
        "clusters": selected_segs,
        "states": selected_states,
        "categories": selected_cats,
        "date_range": date_range,
        "revenue_range": rev_range,
        "clv_range": clv_range
    }


def render_search_input() -> str:
    """Renders universal search input bar."""
    return st.text_input("🔍 Global Search (Customer ID, Order ID, Seller ID, Product ID)", "", placeholder="Enter Customer ID, Order ID, Seller ID, or Product ID...")


def render_customer_search_input() -> str:
    """Renders customer search bar (Customer ID, City, State)."""
    return st.text_input("🔍 Search Customer Intelligence (by Customer ID, City, or State)", "", placeholder="Enter Customer ID, City (e.g. Sao Paulo), or State (e.g. SP)...", key="cust_search_query_input")


def render_segmentation_search_input() -> str:
    """Renders segmentation search bar (Customer ID, Cluster Name, Persona)."""
    return st.text_input("🔍 Search Segmentation Intelligence (by Customer ID, Cluster Name, or Persona)", "", placeholder="Enter Customer ID, Cluster Name (e.g. Cluster 0), or Persona (e.g. VIP Power Buyers)...", key="seg_search_query_input")


def render_churn_search_input() -> str:
    """Renders churn search bar (Customer ID, Customer Name, Email)."""
    return st.text_input("🔍 Search Churn Risk Intelligence (by Customer ID, Name, or Email)", "", placeholder="Enter Customer ID, Name, or Email...", key="churn_search_query_input")


def render_clv_search_input() -> str:
    """Renders CLV search bar (Customer ID, Customer Name, Customer Segment)."""
    return st.text_input("🔍 Search CLV & Revenue Intelligence (by Customer ID, Name, or Segment)", "", placeholder="Enter Customer ID, Name, or Segment...", key="clv_search_query_input")


def render_recommendation_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders interactive sidebar filter controls for AI Recommendation Engine Dashboard."""
    options = filter_options or {}

    st.sidebar.markdown("### 🤖 Recommendation Filters")

    with st.sidebar.expander("🎯 Customer Segment & CLV Tier", expanded=True):
        seg_opts = options.get("customer_segments", ["Champions", "Loyal Customers", "VIP Power Buyers", "At Risk", "Hibernating"])
        selected_segs = st.multiselect("Customer Segment", options=seg_opts, default=[], key="rec_filter_segs")

        clv_opts = options.get("clv_tiers", ["Platinum", "Gold", "Silver", "Bronze"])
        selected_clv = st.multiselect("CLV Tier", options=clv_opts, default=[], key="rec_filter_clv")

    with st.sidebar.expander("⚠️ Churn Risk Tier", expanded=False):
        churn_opts = options.get("churn_risks", ["Critical Risk", "High Risk", "Medium Risk", "Low Risk", "Very Low Risk"])
        selected_churn = st.multiselect("Churn Risk", options=churn_opts, default=[], key="rec_filter_churn")

    with st.sidebar.expander("🛍️ Product Category & Type", expanded=False):
        cat_opts = options.get("categories", ["health_beauty", "bed_bath_table", "sports_leisure", "computers_accessories", "furniture_decor"])
        selected_cats = st.multiselect("Product Category", options=cat_opts, default=[], key="rec_filter_cats")

        type_opts = ["Personalized AI", "Recommended For You", "Similar Product", "Frequently Bought Together", "Cross-Sell", "Upsell", "Trending"]
        selected_types = st.multiselect("Recommendation Type", options=type_opts, default=[], key="rec_filter_types")

    with st.sidebar.expander("⭐ Ratings & Thresholds", expanded=False):
        min_rating = st.slider("Minimum Product Rating", min_value=1.0, max_value=5.0, value=1.0, step=0.1, key="rec_filter_min_rating")
        min_score = st.slider("Minimum Hybrid Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="rec_filter_min_score")

    st.sidebar.markdown("---")

    return {
        "customer_segments": selected_segs,
        "clv_tiers": selected_clv,
        "churn_risks": selected_churn,
        "categories": selected_cats,
        "recommendation_types": selected_types,
        "min_rating": min_rating,
        "min_score": min_score
    }


def render_recommendation_search_input() -> str:
    """Renders recommendation search bar (Customer ID, Product ID, Product Category)."""
    return st.text_input("🔍 Search Recommendation Engine (by Customer ID, Product ID, or Product Category)", "", placeholder="Enter Customer ID, Product ID (e.g. PROD_...), or Category...", key="rec_search_query_input")


def render_mba_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders interactive sidebar filter controls for Market Basket Analysis Dashboard."""
    options = filter_options or {}

    st.sidebar.markdown("### 🛒 Market Basket Filters")

    with st.sidebar.expander("📅 Date Range Filter", expanded=True):
        date_range = st.date_input("Select Transaction Period", [], key="mba_filter_date_range")

    with st.sidebar.expander("🛍️ Product Category", expanded=False):
        cat_opts = options.get("categories", ["health_beauty", "bed_bath_table", "sports_leisure", "computers_accessories", "furniture_decor", "housewares"])
        selected_cats = st.multiselect("Product Category", options=cat_opts, default=[], key="mba_filter_cats")

    with st.sidebar.expander("🎯 Customer Segment", expanded=False):
        seg_opts = options.get("customer_segments", ["VIP Power Buyers", "Loyal Frequenters", "At-Risk High Rollers", "New Customers"])
        selected_segs = st.multiselect("Customer Segment", options=seg_opts, default=[], key="mba_filter_segs")

    with st.sidebar.expander("⚙️ Association Thresholds", expanded=False):
        min_supp = st.slider("Minimum Support", min_value=0.0, max_value=0.1, value=0.001, step=0.001, format="%.3f", key="mba_filter_min_supp")
        min_conf = st.slider("Minimum Confidence", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="mba_filter_min_conf")
        min_lift = st.slider("Minimum Lift Score", min_value=1.0, max_value=10.0, value=1.0, step=0.2, key="mba_filter_min_lift")

    st.sidebar.markdown("---")

    return {
        "date_range": date_range,
        "categories": selected_cats,
        "customer_segments": selected_segs,
        "min_support": min_supp,
        "min_confidence": min_conf,
        "min_lift": min_lift
    }


def render_mba_search_input() -> str:
    """Renders MBA search bar (Product Name, Product ID, Category)."""
    return st.text_input("🔍 Search Market Basket (by Product ID, Category, or Bundle)", "", placeholder="Enter Product ID (e.g. PROD_...), Category (e.g. health_beauty), or Bundle...", key="mba_search_query_input")


def render_mlops_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders interactive sidebar filter controls for MLOps & AI Governance Dashboard."""
    options = filter_options or {}

    st.sidebar.markdown("### 🛡️ MLOps Filters")

    with st.sidebar.expander("📦 Model & Version", expanded=True):
        m_opts = options.get("models", ["CustomerSegmentation", "ChurnClassifier", "CLVRegressor", "HybridRecommender", "MarketBasketMining"])
        selected_models = st.multiselect("Select Model", options=m_opts, default=[], key="mlops_filter_models")

        v_opts = options.get("versions", ["v1.0", "v1.1", "v2.0"])
        selected_versions = st.multiselect("Select Version", options=v_opts, default=[], key="mlops_filter_versions")

    with st.sidebar.expander("🟢 Health & Drift Status", expanded=False):
        status_opts = ["Active", "Archived", "Staging"]
        selected_status = st.multiselect("Deployment Status", options=status_opts, default=[], key="mlops_filter_status")

        drift_opts = ["No Drift", "Drift Detected"]
        selected_drift = st.multiselect("Drift Status", options=drift_opts, default=[], key="mlops_filter_drift")

    with st.sidebar.expander("📅 Audit Event Period", expanded=False):
        date_range = st.date_input("Audit Date Range", [], key="mlops_filter_date_range")

    st.sidebar.markdown("---")

    return {
        "models": selected_models,
        "versions": selected_versions,
        "status": selected_status,
        "drift_status": selected_drift,
        "date_range": date_range
    }


def render_mlops_search_input() -> str:
    """Renders MLOps search bar (Model Name, Version, Experiment ID, Audit Event)."""
    return st.text_input("🔍 Search MLOps Registry & Audit Logs", "", placeholder="Enter Model Name, Version (e.g. v1.0), Run ID, or Audit Event...", key="mlops_search_query_input")


def render_reports_filter_panel(filter_options: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Renders global filter panel for Enterprise Reports & Export Center."""
    options = filter_options or {}

    st.sidebar.markdown("### 📄 Global Report Filters")

    with st.sidebar.expander("📅 Reporting Date Range", expanded=True):
        date_range = st.date_input("Select Period", [], key="rep_filter_date_range")

    with st.sidebar.expander("🗺️ State & Geography", expanded=False):
        state_opts = options.get("states", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA"])
        selected_states = st.multiselect("State", options=state_opts, default=[], key="rep_filter_states")

    with st.sidebar.expander("🎯 Customer Segment & CLV Tier", expanded=False):
        seg_opts = options.get("customer_segments", ["Champions", "Loyal Customers", "VIP Power Buyers", "At Risk", "Hibernating"])
        selected_segs = st.multiselect("Customer Segment", options=seg_opts, default=[], key="rep_filter_segs")

        clv_opts = options.get("clv_tiers", ["Platinum", "Gold", "Silver", "Bronze"])
        selected_clv = st.multiselect("CLV Tier", options=clv_opts, default=[], key="rep_filter_clv")

    with st.sidebar.expander("⚠️ Churn Risk Tier", expanded=False):
        churn_opts = options.get("churn_risks", ["Critical Risk", "High Risk", "Medium Risk", "Low Risk", "Very Low Risk"])
        selected_churn = st.multiselect("Churn Risk", options=churn_opts, default=[], key="rep_filter_churn")

    with st.sidebar.expander("🛍️ Product Category", expanded=False):
        cat_opts = options.get("categories", ["health_beauty", "bed_bath_table", "sports_leisure", "computers_accessories", "furniture_decor"])
        selected_cats = st.multiselect("Product Category", options=cat_opts, default=[], key="rep_filter_cats")

    st.sidebar.markdown("---")

    return {
        "date_range": date_range,
        "states": selected_states,
        "customer_segments": selected_segs,
        "clv_tiers": selected_clv,
        "churn_risks": selected_churn,
        "categories": selected_cats
    }


def render_reports_search_input() -> str:
    """Renders Reports search bar (Report Name, Category, Format)."""
    return st.text_input("🔍 Search Reports Catalog (by Name, Category, or Format)", "", placeholder="Enter Report Name, Category (e.g. Executive, AI), or Format (PDF, Excel)...", key="rep_search_query_input")




