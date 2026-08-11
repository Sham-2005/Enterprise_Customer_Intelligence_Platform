"""
Seller & Merchant Performance Dashboard Page for ECIP.
Displays seller leaderboard, merchant revenue distribution, rating score, and seller location density.
"""

import pandas as pd
import streamlit as st
from dashboard.components.charts import (
    create_bar_chart, create_pie_chart, create_box_plot
)
from dashboard.utils.exporter import render_export_buttons

def render_seller_analytics_page(master_df: pd.DataFrame):
    st.title("🏪 Seller & Merchant Intelligence")
    st.markdown("Merchant leaderboard rankings, seller state distribution, delivery performance, and rating metrics.")
    st.markdown("---")

    if "seller_id" in master_df.columns:
        sellers_df = master_df.groupby("seller_id").agg(
            total_revenue=("price", "sum"),
            total_orders=("order_id", "nunique"),
            avg_seller_rating=("avg_review_score", "mean"),
            seller_state=("seller_state", "first")
        ).reset_index()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Active Sellers", f"{len(sellers_df):,}")
        with c2:
            avg_seller_rev = sellers_df["total_revenue"].mean()
            st.metric("Avg Revenue per Seller", f"${avg_seller_rev:,.2f}")
        with c3:
            avg_merchant_rating = sellers_df["avg_seller_rating"].mean()
            st.metric("Avg Seller Rating", f"⭐ {avg_merchant_rating:.2f}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            top_sellers = sellers_df.sort_values(by="total_revenue", ascending=False).head(10)
            fig_top_sellers = create_bar_chart(
                top_sellers, "total_revenue", "seller_id", "Top 10 Sellers by Gross Revenue ($ USD)", orientation="h"
            )
            st.plotly_chart(fig_top_sellers, use_container_width=True)

        with col2:
            seller_states = sellers_df["seller_state"].value_counts().head(8).reset_index()
            seller_states.columns = ["seller_state", "seller_count"]
            fig_seller_states = create_pie_chart(
                seller_states, "seller_state", "seller_count", "Seller Geographic Distribution by State"
            )
            st.plotly_chart(fig_seller_states, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Export Merchant Performance Data")
        render_export_buttons(sellers_df, "seller_analytics")
