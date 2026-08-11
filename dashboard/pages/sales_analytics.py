"""
Sales Analytics Dashboard Page for ECIP.
Displays revenue breakdowns by category, state, seller, seasonal trends, and payment methods.
"""

import pandas as pd
import streamlit as st
from dashboard.components.charts import (
    create_line_chart, create_bar_chart, create_pie_chart, create_treemap
)
from dashboard.utils.exporter import render_export_buttons

def render_sales_analytics_page(master_df: pd.DataFrame):
    st.title("📈 Commercial Sales & Revenue Analytics")
    st.markdown("Detailed breakdown of sales dynamics, seasonal patterns, payment breakdown, and revenue geography.")
    st.markdown("---")

    # Sales Trends by Season / Payment Type
    col1, col2 = st.columns(2)

    with col1:
        if "preferred_payment_type" in master_df.columns:
            payment_df = master_df.groupby("preferred_payment_type")["price"].sum().reset_index()
            fig_payment = create_pie_chart(
                payment_df, "preferred_payment_type", "price", "Revenue by Payment Method"
            )
            st.plotly_chart(fig_payment, use_container_width=True)

    with col2:
        if "season" in master_df.columns:
            season_df = master_df.groupby("season")["price"].sum().reset_index()
            fig_season = create_bar_chart(
                season_df, "season", "price", "Revenue Distribution by Season"
            )
            st.plotly_chart(fig_season, use_container_width=True)

    # Revenue Treemap by Category and Customer State
    st.markdown("---")
    st.markdown("### 🗺️ Revenue Hierarchy (Category & State)")
    if "product_category_name_english" in master_df.columns and "customer_state" in master_df.columns:
        top_cats = master_df["product_category_name_english"].value_counts().head(10).index
        tree_df = master_df[master_df["product_category_name_english"].isin(top_cats)]
        fig_tree = create_treemap(
            tree_df, ["product_category_name_english", "customer_state"], "price", "Revenue Breakdown by Category & State"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📥 Export Commercial Sales Data")
    render_export_buttons(master_df.head(100), "sales_analytics")
