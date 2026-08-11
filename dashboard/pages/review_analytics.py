"""
Review & CSAT Analytics Dashboard Page for ECIP.
Displays customer rating distributions, monthly review trends, and Radar category comparison.
"""

import pandas as pd
import streamlit as st
from dashboard.components.charts import (
    create_bar_chart, create_pie_chart, create_radar_chart, create_line_chart
)
from dashboard.utils.exporter import render_export_buttons

def render_review_analytics_page(master_df: pd.DataFrame):
    st.title("⭐ Customer Review & CSAT Analytics")
    st.markdown("Rating distributions, monthly review sentiment trends, and multi-category radar benchmarks.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        if "avg_review_score" in master_df.columns:
            avg_score = master_df["avg_review_score"].mean()
            st.metric("Overall CSAT Rating", f"⭐ {avg_score:.2f} / 5.0")
    with c2:
        if "avg_review_score" in master_df.columns:
            five_star = (master_df["avg_review_score"] == 5).mean() * 100
            st.metric("5-Star Review Ratio", f"{five_star:.1f}%")
    with c3:
        if "avg_review_score" in master_df.columns:
            low_reviews = (master_df["avg_review_score"] <= 2).mean() * 100
            st.metric("Negative Review Ratio (≤ 2)", f"{low_reviews:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if "avg_review_score" in master_df.columns:
            ratings_df = master_df["avg_review_score"].round().value_counts().reset_index()
            ratings_df.columns = ["review_score", "count"]
            fig_ratings = create_bar_chart(
                ratings_df, "review_score", "count", "Review Score Distribution (1 - 5 Stars)"
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

    with col2:
        if "product_category_name_english" in master_df.columns and "avg_review_score" in master_df.columns:
            top_cats = master_df.groupby("product_category_name_english")["avg_review_score"].mean().head(6)
            categories = list(top_cats.index)
            scores = list(top_cats.values)
            fig_radar = create_radar_chart(categories, scores, "Category Review Score Radar Benchmark")
            st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📥 Export Review Analytics Data")
    render_export_buttons(master_df.head(100), "review_analytics")
