"""
Product Analytics Dashboard Page for ECIP.
Displays best/worst selling products, Pareto 80/20 category revenue, product ratings, and popularity.
"""

import pandas as pd
import streamlit as st
from dashboard.components.charts import (
    create_bar_chart, create_pareto_chart, create_pie_chart, create_treemap
)
from dashboard.utils.exporter import render_export_buttons

def render_product_analytics_page(master_df: pd.DataFrame):
    st.title("📦 Product Catalog Performance Analytics")
    st.markdown("Product sales velocity, category dominance, Pareto 80/20 revenue concentration, and rating metrics.")
    st.markdown("---")

    if "product_category_name_english" in master_df.columns and "price" in master_df.columns:
        # Pareto Chart (80/20 rule analysis)
        st.markdown("### ⚖️ Pareto 80/20 Revenue Concentration by Category")
        fig_pareto = create_pareto_chart(
            master_df, "product_category_name_english", "price", "Pareto Analysis: Top Categories vs Cumulative Revenue %"
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if "product_id" in master_df.columns:
            top_products = master_df.groupby("product_id").agg(
                units_sold=("order_item_id", "count"),
                revenue=("price", "sum")
            ).sort_values(by="revenue", ascending=False).head(10).reset_index()

            fig_top = create_bar_chart(
                top_products, "revenue", "product_id", "Top 10 Products by Revenue ($ USD)", orientation="h"
            )
            st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        if "product_category_name_english" in master_df.columns:
            cat_ratings = master_df.groupby("product_category_name_english").agg(
                avg_rating=("avg_review_score", "mean"),
                total_sales=("order_item_id", "count")
            ).sort_values(by="avg_rating", ascending=False).head(10).reset_index()

            fig_ratings = create_bar_chart(
                cat_ratings, "avg_rating", "product_category_name_english", "Top 10 Highest Rated Product Categories", orientation="h"
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📥 Export Product Performance Metrics")
    render_export_buttons(master_df.head(100), "product_analytics")
