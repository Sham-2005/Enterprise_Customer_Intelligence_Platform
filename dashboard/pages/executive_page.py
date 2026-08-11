"""
Executive Dashboard Page UI Layout for ECIP.
Fully integrated with ExecutiveDashboardBackend service layer.
Connects real-time dataset calculations for KPIs, Revenue Trends, Customer Growth,
Category Treemaps, Geospatial State Maps, Payment Methods, Order Status, Top Products,
Ratings Histogram, Automated Insights, Executive Summary, Global Search, and File Exports.
"""

import streamlit as st
import pandas as pd
from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb
from dashboard.components.kpi_cards import render_glass_kpi_card
from dashboard.components.filters import render_global_filter_panel, render_search_input
from dashboard.components.charts import (
    create_line_chart,
    create_customer_growth_chart,
    create_treemap_chart,
    create_state_map_chart,
    create_donut_chart,
    create_bar_chart,
    create_histogram_chart
)
from backend.dashboard.executive_backend import executive_backend
from utils.logger import setup_logger

logger = setup_logger("ECIP.ExecutivePage")

def render_executive_dashboard_layout():
    """Renders complete Executive Dashboard UI Page with Backend Integration."""
    render_top_header("Executive Intelligence Dashboard")
    render_breadcrumb(["Home", "Executive Portal"])

    # 1. Render Sidebar Filter Controls
    filter_options = executive_backend.get_filter_options()
    filters = render_global_filter_panel(filter_options)

    # 2. Render Global Search Input Bar
    st.markdown("### 🔍 Enterprise Global Search")
    search_query = render_search_input()

    if search_query.strip():
        search_result = executive_backend.execute_global_search(search_query)
        _render_search_results(search_result)
        st.markdown("---")

    # Granularity Selector for Revenue Trend Chart
    rev_granularity = st.sidebar.radio(
        "📈 Revenue Trend Granularity",
        options=["Monthly", "Weekly", "Daily"],
        index=0,
        key="exec_rev_granularity"
    )

    # 3. Load Backend Payload
    with st.spinner("Compiling C-suite executive analytics from processed backend datasets..."):
        payload = executive_backend.get_dashboard_payload(
            date_range=filters.get("date_range"),
            states=filters.get("states"),
            categories=filters.get("categories"),
            sellers=filters.get("sellers"),
            payment_methods=filters.get("payment_methods"),
            customer_segments=filters.get("customer_segments"),
            revenue_granularity=rev_granularity
        )

    # 4. Check Dataset Diagnostics & Banners
    _render_datasets_status_banner(payload.get("datasets_status", {}))

    # 5. Render 8 KPI Cards Grid Row
    kpis = payload.get("kpis", {})
    _render_kpi_grid_section(kpis)

    st.markdown("---")

    # 6. Render Executive Summary Box
    summary_text = payload.get("executive_summary", "")
    st.info(summary_text)

    st.markdown("---")

    # 7. Main Visual Charts Grid Layout
    charts_data = payload.get("charts", {})

    # Row 1: Revenue Trend Line Chart & Category Treemap
    c1, c2 = st.columns([1.6, 1.2])
    with c1:
        rev_df = charts_data.get("revenue_trend", pd.DataFrame())
        fig_trend = create_line_chart(rev_df, x="Period", y="Revenue", title=f"Gross Revenue Trajectory ({rev_granularity})")
        st.plotly_chart(fig_trend, use_container_width=True)
    with c2:
        tree_df = charts_data.get("category_treemap", pd.DataFrame())
        fig_tree = create_treemap_chart(tree_df, path_col="Category", values_col="Revenue", title="Revenue by Product Category (Treemap)")
        st.plotly_chart(fig_tree, use_container_width=True)

    # Row 2: Customer Growth Chart & Geospatial State Map
    c3, c4 = st.columns([1.4, 1.4])
    with c3:
        growth_df = charts_data.get("customer_growth", pd.DataFrame())
        fig_growth = create_customer_growth_chart(growth_df, title="Customer Base Growth (New vs Returning)")
        st.plotly_chart(fig_growth, use_container_width=True)
    with c4:
        state_df = charts_data.get("state_map", pd.DataFrame())
        fig_map = create_state_map_chart(state_df, title="Revenue & Customer Distribution by State (Map)")
        st.plotly_chart(fig_map, use_container_width=True)

    # Row 3: Payment Method Distribution & Order Status
    c5, c6 = st.columns([1.2, 1.4])
    with c5:
        pmt_df = charts_data.get("payment_distribution", pd.DataFrame())
        fig_donut = create_donut_chart(pmt_df, names="Payment_Method", values="Total_Revenue", title="Payment Method Share (%)")
        st.plotly_chart(fig_donut, use_container_width=True)
    with c6:
        status_df = charts_data.get("order_status", pd.DataFrame())
        fig_status = create_bar_chart(status_df, x="Order_Status", y="Count", title="Order Fulfillment Status Distribution")
        st.plotly_chart(fig_status, use_container_width=True)

    # Row 4: Top Selling Products & Customer Ratings Histogram
    c7, c8 = st.columns([1.4, 1.2])
    with c7:
        top_prod_df = charts_data.get("top_products", pd.DataFrame())
        fig_prod = create_bar_chart(top_prod_df, x="Revenue", y="Product_ID_Short", horizontal=True, title="Top 10 Selling Products ($ Revenue)")
        st.plotly_chart(fig_prod, use_container_width=True)
    with c8:
        rating_df = charts_data.get("ratings_histogram", pd.DataFrame())
        fig_hist = create_histogram_chart(rating_df, x="Star_Rating", y="Count", title="Customer Feedback CSAT Rating Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # 8. Render Recent Business Insights Cards
    st.markdown("### 💡 Recent Business Intelligence Insights")
    insights = payload.get("recent_insights", [])
    _render_insights_grid(insights)

    st.markdown("---")

    # 9. Render Export Data & Reports Section
    _render_export_reports_section(
        filtered_master_df=payload.get("filtered_master_df", pd.DataFrame()),
        kpis=kpis,
        summary_text=summary_text
    )


def _render_kpi_grid_section(kpis: dict):
    """Renders the 8 Executive KPI cards in 2 rows of 4 columns."""
    st.markdown("### 🌐 Executive Performance Indicators (KPIs)")

    kpi_keys = [
        "total_revenue", "total_orders", "total_customers", "avg_order_value",
        "avg_rating", "retention_rate", "monthly_revenue_growth", "business_health_score"
    ]

    # Row 1 (Top 4 KPIs)
    r1 = st.columns(4)
    for idx in range(4):
        k = kpi_keys[idx]
        m = kpis.get(k, {})
        with r1[idx]:
            render_glass_kpi_card(
                title=m.get("title", k.replace("_", " ").title()),
                value=m.get("value", "N/A"),
                change_pct=m.get("change_pct", "↑ 0.0%"),
                is_positive=m.get("is_positive", True),
                subtext=m.get("subtext", "vs prev period"),
                icon=m.get("icon", "📊"),
                badge_type="green" if m.get("is_positive", True) else "red",
                last_updated=m.get("last_updated", "")
            )

    # Row 2 (Bottom 4 KPIs)
    r2 = st.columns(4)
    for idx in range(4, 8):
        k = kpi_keys[idx]
        m = kpis.get(k, {})
        with r2[idx - 4]:
            render_glass_kpi_card(
                title=m.get("title", k.replace("_", " ").title()),
                value=m.get("value", "N/A"),
                change_pct=m.get("change_pct", "↑ 0.0%"),
                is_positive=m.get("is_positive", True),
                subtext=m.get("subtext", "vs prev period"),
                icon=m.get("icon", "📊"),
                badge_type="green" if m.get("is_positive", True) else "red",
                last_updated=m.get("last_updated", "")
            )


def _render_insights_grid(insights: list):
    """Renders business insight cards in responsive grid columns."""
    if not insights:
        return

    cols = st.columns(min(3, len(insights)))
    for idx, ins in enumerate(insights):
        col_idx = idx % min(3, len(insights))
        with cols[col_idx]:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 12px; padding: 14px;">
                    <div style="font-size: 22px; margin-bottom: 4px;">{ins.get('icon', '💡')}</div>
                    <div style="font-weight: 600; font-size: 14px; color: #f8fafc;">{ins.get('title')}</div>
                    <div style="font-size: 18px; font-weight: 700; color: #38bdf8; margin: 4px 0;">{ins.get('value')}</div>
                    <div style="font-size: 12px; color: #94a3b8;">{ins.get('description')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_search_results(result: dict):
    """Renders structured search results container."""
    if not result.get("has_match"):
        st.warning(result.get("summary", {}).get("message", f"No matching entities found for query: '{result.get('query')}'"))
        return

    m_type = result.get("match_type", "Entity")
    summary = result.get("summary", {})
    records = result.get("records", pd.DataFrame())

    st.success(f"🎯 **Found Matching {m_type} Entity**: {result.get('query')}")

    with st.expander(f"📌 {m_type} Summary & Related Transactions", expanded=True):
        m_cols = st.columns(len(summary))
        for idx, (k, v) in enumerate(summary.items()):
            with m_cols[idx]:
                st.metric(label=k, value=str(v))

        if not records.empty:
            st.markdown("#### Related Transactions & History")
            st.dataframe(records.head(20), use_container_width=True)


def _render_datasets_status_banner(status: dict):
    """Renders status info for missing or loaded datasets."""
    missing = [k for k, v in status.items() if not v.get("available")]
    if missing:
        st.warning(f"⚠️ **Notice**: The following processed datasets were not found in pipeline output: `{', '.join(missing)}`. Fallback datasets or raw merged files are active.")


def _render_export_reports_section(filtered_master_df: pd.DataFrame, kpis: dict, summary_text: str):
    """Renders CSV, Excel, and PDF export download buttons."""
    st.markdown("### 📥 Reports & Data Export Portal")
    st.markdown("Download current filtered executive dataset in CSV, Excel, or formatted PDF report formats.")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_bytes, csv_fn, csv_mime = executive_backend.generate_export_file("csv", filtered_master_df)
        st.download_button(
            label="📄 Download CSV Data",
            data=csv_bytes,
            file_name=csv_fn,
            mime=csv_mime,
            use_container_width=True,
            key="btn_export_csv"
        )

    with c2:
        excel_bytes, excel_fn, excel_mime = executive_backend.generate_export_file("excel", filtered_master_df)
        st.download_button(
            label="📊 Download Excel Workbook",
            data=excel_bytes,
            file_name=excel_fn,
            mime=excel_mime,
            use_container_width=True,
            key="btn_export_excel"
        )

    with c3:
        pdf_bytes, pdf_fn, pdf_mime = executive_backend.generate_export_file(
            "pdf",
            filtered_master_df,
            kpi_metrics=kpis,
            summary_text=summary_text
        )
        st.download_button(
            label="📕 Download Executive PDF Report",
            data=pdf_bytes,
            file_name=pdf_fn,
            mime=pdf_mime,
            use_container_width=True,
            key="btn_export_pdf"
        )
