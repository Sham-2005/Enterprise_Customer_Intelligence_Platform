"""
Order & Fulfillment Analytics Dashboard Page for ECIP.
Displays delivery time distributions, shipping cost ratios, freight analysis, and delivery delays.
"""

import pandas as pd
import streamlit as st
from dashboard.components.charts import (
    create_box_plot, create_violin_plot, create_bar_chart, create_pie_chart
)
from dashboard.utils.exporter import render_export_buttons

def render_order_analytics_page(master_df: pd.DataFrame):
    st.title("🚚 Order Fulfillment & Logistics Analytics")
    st.markdown("Delivery duration distributions, estimated vs actual delivery delays, freight cost ratio, and status.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if "delivery_time_days" in master_df.columns:
            avg_deliv = master_df["delivery_time_days"].mean()
            st.metric("Avg Delivery Time", f"{avg_deliv:.1f} Days")
    with c2:
        if "is_delayed" in master_df.columns:
            delay_pct = (master_df["is_delayed"].mean() * 100)
            st.metric("Delayed Delivery Rate", f"{delay_pct:.2f}%")
    with c3:
        if "shipping_cost_ratio" in master_df.columns:
            avg_freight_ratio = (master_df["shipping_cost_ratio"].mean() * 100)
            st.metric("Avg Shipping Cost Ratio", f"{avg_freight_ratio:.1f}%")
    with c4:
        if "freight_value" in master_df.columns:
            total_freight = master_df["freight_value"].sum()
            st.metric("Total Freight Cost", f"${total_freight:,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if "delivery_time_days" in master_df.columns and "customer_state" in master_df.columns:
            top_states = master_df["customer_state"].value_counts().head(8).index
            filtered_states = master_df[master_df["customer_state"].isin(top_states)]
            fig_box = create_box_plot(
                filtered_states, "customer_state", "delivery_time_days", "Delivery Duration (Days) by State"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        if "order_status" in master_df.columns:
            status_df = master_df["order_status"].value_counts().reset_index()
            status_df.columns = ["order_status", "count"]
            fig_pie = create_pie_chart(
                status_df, "order_status", "count", "Order Fulfillment Status Breakdown"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📥 Export Logistics Data")
    render_export_buttons(master_df.head(100), "order_analytics")
