"""
Universal Search & Entity Intelligence Page for ECIP.
Allows instant query lookup by Customer ID, Order ID, Product ID, Seller ID, or Category.
"""

import pandas as pd
import streamlit as st
from dashboard.utils.exporter import render_export_buttons

def render_search_page(master_df: pd.DataFrame, feature_store_df: pd.DataFrame):
    st.title("🔍 Entity Intelligence Search Engine")
    st.markdown("Instantly inspect granular records by Customer ID, Order ID, Product ID, Seller ID, or Category.")
    st.markdown("---")

    search_type = st.selectbox(
        "Select Entity Search Type",
        options=["Customer ID", "Order ID", "Product ID", "Seller ID", "Product Category"]
    )

    query = st.text_input("Enter Query ID or Name...", placeholder="Type query string here...")

    if query:
        query_str = query.strip()
        st.markdown(f"### Results for `{query_str}` under `{search_type}`")

        results_df = pd.DataFrame()

        if search_type == "Customer ID":
            if "customer_unique_id" in feature_store_df.columns:
                results_df = feature_store_df[
                    feature_store_df["customer_unique_id"].astype(str).str.contains(query_str, case=False, na=False)
                ]

        elif search_type == "Order ID":
            if "order_id" in master_df.columns:
                results_df = master_df[
                    master_df["order_id"].astype(str).str.contains(query_str, case=False, na=False)
                ]

        elif search_type == "Product ID":
            if "product_id" in master_df.columns:
                results_df = master_df[
                    master_df["product_id"].astype(str).str.contains(query_str, case=False, na=False)
                ]

        elif search_type == "Seller ID":
            if "seller_id" in master_df.columns:
                results_df = master_df[
                    master_df["seller_id"].astype(str).str.contains(query_str, case=False, na=False)
                ]

        elif search_type == "Product Category":
            if "product_category_name_english" in master_df.columns:
                results_df = master_df[
                    master_df["product_category_name_english"].astype(str).str.contains(query_str, case=False, na=False)
                ]

        if not results_df.empty:
            st.success(f"Found {len(results_df):,} matching record(s).")
            st.dataframe(results_df, use_container_width=True)

            st.markdown("### 📥 Export Search Results")
            render_export_buttons(results_df, f"search_results_{search_type.lower().replace(' ', '_')}")
        else:
            st.warning(f"No records found matching query '{query_str}'. Please verify your search term.")
