"""
Customer Segmentation & RFM Intelligence Page UI Layout for ECIP Phase 13.
Fully integrated with SegmentationService backend.
Provides 8 KPIs, Cluster Overview, Interactive Cluster Explorer, Business Personas Cards,
RFM Dashboard (Heatmaps & Quintiles), 2D/3D PCA Dimensionality Plots, Cluster Comparison Matrix,
Automated Marketing Intelligence Cards, Segmentation Search, and Multi-Format Exports.
"""

import streamlit as st
import pandas as pd
from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb
from dashboard.components.kpi_cards import render_glass_kpi_card
from dashboard.components.filters import render_segmentation_filter_panel, render_segmentation_search_input
from dashboard.components.charts import (
    create_line_chart,
    create_treemap_chart,
    create_donut_chart,
    create_bar_chart,
    create_histogram_chart,
    create_rfm_heatmap,
    create_3d_pca_scatter
)
from backend.services.segmentation_service import segmentation_service
from utils.logger import setup_logger

logger = setup_logger("ECIP.SegmentationPage")

def render_segmentation_layout():
    """Renders complete Customer Segmentation & RFM Intelligence UI Page."""
    render_top_header("Customer Segmentation & RFM Intelligence")
    render_breadcrumb(["Home", "Analytics", "Customer Segmentation"])

    # 1. Sidebar Filters
    filter_opts = segmentation_service.get_filter_options()
    filters = render_segmentation_filter_panel(filter_opts)

    # 2. Segmentation Search Bar
    st.markdown("### 🔍 Segmentation & Persona Search")
    search_query = render_segmentation_search_input()

    if search_query.strip():
        search_res = segmentation_service.search_segmentation_profile(search_query)
        _render_segmentation_search_results(search_res)
        st.markdown("---")

    # 3. Load Backend Payload
    with st.spinner("Compiling unsupervised AI customer clusters, PCA projections, and RFM intelligence..."):
        payload = segmentation_service.get_segmentation_payload(
            date_range=filters.get("date_range"),
            clusters=filters.get("clusters"),
            personas=filters.get("personas"),
            states=filters.get("states"),
            categories=filters.get("categories"),
            revenue_range=filters.get("revenue_range"),
            clv_range=filters.get("clv_range"),
            churn_risk=filters.get("churn_risk")
        )

    # 4. Datasets Status Notice
    _render_datasets_status_notice(payload.get("datasets_status", {}))

    # 5. Render 8 KPI Cards Grid
    kpis = payload.get("kpis", {})
    _render_8_kpi_grid_section(kpis)

    st.markdown("---")

    # 6. Cluster Overview Section
    st.markdown("### 🌐 Cluster Distribution & Revenue Overview")
    _render_cluster_overview_section(payload.get("cluster_overview", {}))

    st.markdown("---")

    # 7. Interactive Cluster Explorer
    st.markdown("### 🔍 Interactive Cluster Explorer")
    _render_interactive_cluster_explorer(payload.get("cluster_overview", {}))

    st.markdown("---")

    # 8. Customer Personas Cards Section
    st.markdown("### 👥 Business Personas & Strategic Blueprints")
    _render_personas_section(payload.get("personas", []))

    st.markdown("---")

    # 9. RFM Analytics Dashboard Section
    st.markdown("### 📊 RFM Quintiles & Matrix Dashboard")
    _render_rfm_dashboard_section(payload.get("rfm_dashboard", {}))

    st.markdown("---")

    # 10. PCA Cluster Dimensionality Plots (2D & 3D)
    st.markdown("### 🌀 PCA Dimensionality Reduction Cluster Plots")
    _render_pca_section(payload.get("pca_data", pd.DataFrame()))

    st.markdown("---")

    # 11. Side-by-Side Cluster Comparison Matrix
    st.markdown("### ⚖️ Side-by-Side Cluster Metrics Comparison")
    _render_cluster_comparison_section(payload.get("cluster_comparison", pd.DataFrame()))

    st.markdown("---")

    # 12. Automated Marketing Intelligence Recommendations
    st.markdown("### 💡 Strategic Marketing Intelligence")
    _render_marketing_recommendations_section(payload.get("marketing_recommendations", []))

    st.markdown("---")

    # 13. Reports Export Portal
    _render_export_portal(
        filtered_fs_df=payload.get("filtered_fs_df", pd.DataFrame()),
        kpis=kpis
    )


def _render_8_kpi_grid_section(kpis: dict):
    """Renders 8 Segmentation KPI cards in 2 rows of 4 columns."""
    st.markdown("### 🌐 Segmentation Performance Indicators (KPIs)")

    kpi_keys = [
        "total_segments", "total_customers_clustered", "vip_customers", "loyal_customers",
        "at_risk_customers", "avg_cluster_revenue", "avg_rfm_score", "largest_customer_segment"
    ]

    # Row 1 (4 Cards)
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
                icon=m.get("icon", "🧩"),
                badge_type="green" if m.get("is_positive", True) else "red",
                last_updated=m.get("last_updated", "")
            )

    # Row 2 (4 Cards)
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
                icon=m.get("icon", "🧩"),
                badge_type="green" if m.get("is_positive", True) else "red",
                last_updated=m.get("last_updated", "")
            )


def _render_cluster_overview_section(cluster_overview: dict):
    """Renders Cluster Distribution Donut, Revenue Treemap, and Customer Count Bar Chart."""
    dist_df = cluster_overview.get("distribution", pd.DataFrame())
    tree_df = cluster_overview.get("revenue_treemap", pd.DataFrame())

    c1, c2 = st.columns([1.4, 1.4])
    with c1:
        fig_donut = create_donut_chart(dist_df, names="Cluster_Name", values="Customer_Count", title="Customer Share by Cluster (%)")
        st.plotly_chart(fig_donut, use_container_width=True)
    with c2:
        fig_tree = create_treemap_chart(tree_df, path_col="Cluster_Name", values_col="Total_Revenue", title="Revenue Share by Cluster Treemap ($)")
        st.plotly_chart(fig_tree, use_container_width=True)


def _render_interactive_cluster_explorer(cluster_overview: dict):
    """Renders interactive dropdown for selecting a cluster and viewing comprehensive metrics."""
    averages_df = cluster_overview.get("averages", pd.DataFrame())
    cluster_names = averages_df["Cluster_Name"].tolist() if not averages_df.empty and "Cluster_Name" in averages_df.columns else ["Champions", "Loyal Customers", "At Risk", "Hibernating"]

    selected_cluster = st.selectbox(
        "Select Cluster to Inspect",
        options=cluster_names,
        index=0,
        key="select_cluster_explorer_dropdown"
    )

    details = segmentation_service.get_selected_cluster_details(selected_cluster)

    st.markdown(f"#### 📌 Deep-Dive Profile: **{selected_cluster}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers", details.get("number_of_customers"))
        st.metric("Total Cluster Revenue", details.get("total_revenue"))
    with c2:
        st.metric("Average Orders", details.get("avg_orders"))
        st.metric("Average Basket Size", details.get("avg_basket_size"))
    with c3:
        st.metric("Average CSAT Rating", details.get("avg_rating"))
        st.metric("Average Predicted CLV", details.get("avg_clv"))
    with c4:
        st.metric("Average Churn Risk", details.get("avg_churn_probability"))
        st.metric("Preferred Payment", details.get("preferred_payment_method"))

    st.markdown(f"**Top Product Categories**: `{details.get('top_product_categories')}` | **Top States**: `{details.get('top_states')}`")


def _render_personas_section(personas: list):
    """Renders Business Persona cards in grid columns."""
    if not personas:
        return

    cols = st.columns(2)
    for idx, p in enumerate(personas):
        col_idx = idx % 2
        with cols[col_idx]:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 16px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">{p.get('icon')} {p.get('persona_title')}</div>
                        <span class="badge-cyan">{p.get('revenue_contribution')}</span>
                    </div>
                    <div style="font-size: 13px; color: #cbd5e1; margin: 8px 0;">{p.get('description')}</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">
                        <b>Buying Behavior:</b> {p.get('buying_behavior')}<br>
                        <b>Frequency:</b> {p.get('purchase_frequency')}<br>
                        <b>Marketing Recommendation:</b> <span style="color: #38bdf8;">{p.get('marketing_recommendation')}</span><br>
                        <b>Retention Strategy:</b> <span style="color: #34d399;">{p.get('retention_strategy')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_rfm_dashboard_section(rfm_dashboard: dict):
    """Renders RFM Quintiles, RFM Heatmap, and RFM Segment breakdown."""
    quintiles = rfm_dashboard.get("quintiles", {})
    heatmap_matrix = rfm_dashboard.get("heatmap_matrix", pd.DataFrame())
    segments_df = rfm_dashboard.get("segments_distribution", pd.DataFrame())

    c1, c2 = st.columns([1.5, 1.3])
    with c1:
        fig_heat = create_rfm_heatmap(heatmap_matrix, title="2D RFM Heatmap (Recency Score vs Frequency Score)")
        st.plotly_chart(fig_heat, use_container_width=True)
    with c2:
        fig_seg = create_bar_chart(segments_df, x="Customer_Count", y="RFM_Segment", horizontal=True, title="RFM Segment Breakdown (Accounts Count)")
        st.plotly_chart(fig_seg, use_container_width=True)

    r_df = quintiles.get("recency", pd.DataFrame())
    f_df = quintiles.get("frequency", pd.DataFrame())
    m_df = quintiles.get("monetary", pd.DataFrame())

    q1, q2, q3 = st.columns(3)
    with q1:
        fig_r = create_bar_chart(r_df, x="Quintile_Score", y="Customer_Count", title="Recency Scores (1-5)")
        st.plotly_chart(fig_r, use_container_width=True)
    with q2:
        fig_f = create_bar_chart(f_df, x="Quintile_Score", y="Customer_Count", title="Frequency Scores (1-5)")
        st.plotly_chart(fig_f, use_container_width=True)
    with q3:
        fig_m = create_bar_chart(m_df, x="Quintile_Score", y="Customer_Count", title="Monetary Scores (1-5)")
        st.plotly_chart(fig_m, use_container_width=True)


def _render_pca_section(pca_df: pd.DataFrame):
    """Renders 2D & 3D PCA Scatter plots."""
    if pca_df.empty:
        st.info("No PCA coordinate dataset available.")
        return

    tab1, tab2 = st.tabs(["🌐 3D PCA Scatter Plot", "📊 2D PCA Cluster Plot"])

    with tab1:
        fig_3d = create_3d_pca_scatter(pca_df, x="PC1", y="PC2", z="PC3", color="Cluster_Name", title="3D PCA Customer Cluster Visualization")
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        fig_2d = create_bar_chart(pca_df.head(20), x="PC1", y="PC2", title="2D PCA Dimensionality Reduction Projection")
        st.plotly_chart(fig_2d, use_container_width=True)


def _render_cluster_comparison_section(comp_df: pd.DataFrame):
    """Renders side-by-side cluster metrics comparison table."""
    if comp_df.empty:
        st.info("No cluster comparison matrix available.")
        return

    st.dataframe(comp_df, use_container_width=True)


def _render_marketing_recommendations_section(recommendations: list):
    """Renders 6 Marketing Intelligence recommendation cards."""
    if not recommendations:
        return

    cols = st.columns(min(3, len(recommendations)))
    for idx, rec in enumerate(recommendations):
        col_idx = idx % min(3, len(recommendations))
        with cols[col_idx]:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 12px; padding: 14px;">
                    <div style="font-size: 22px; margin-bottom: 4px;">{rec.get('icon', '💡')}</div>
                    <div style="font-weight: 600; font-size: 14px; color: #f8fafc;">{rec.get('title')}</div>
                    <div style="font-size: 12px; color: #38bdf8; font-weight: 700; margin: 2px 0;">{rec.get('target_segment')} ({rec.get('customer_count')})</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;"><b>Action:</b> {rec.get('action_plan')}</div>
                    <div style="font-size: 11px; color: #34d399; margin-top: 4px;"><b>Impact:</b> {rec.get('expected_impact')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_segmentation_search_results(result: dict):
    """Renders Segmentation Search Profile card."""
    if not result.get("has_match"):
        st.warning(result.get("message", f"No segmentation result for query '{result.get('query')}'"))
        return

    m_type = result.get("match_type", "Entity")
    profile = result.get("profile", {})
    records = result.get("records", pd.DataFrame())

    st.success(f"🎯 **Found Matching {m_type}**: {result.get('query')}")

    with st.expander(f"📌 {m_type} Detailed Intelligence Profile", expanded=True):
        p_cols = st.columns(min(4, len(profile)))
        for idx, (k, v) in enumerate(profile.items()):
            col_idx = idx % min(4, len(profile))
            with p_cols[col_idx]:
                st.metric(label=k, value=str(v))

        if not records.empty:
            st.markdown("#### Sample Customers in Segment")
            st.dataframe(records.head(30), use_container_width=True)


def _render_datasets_status_notice(status: dict):
    """Notice banner for dataset availability."""
    missing = [k for k, v in status.items() if not v.get("available")]
    if missing:
        st.warning(f"⚠️ **Notice**: Pipeline datasets missing: `{', '.join(missing)}`. Fallback datasets or raw merged files are active.")


def _render_export_portal(filtered_fs_df: pd.DataFrame, kpis: dict):
    """Renders export buttons for CSV, Excel, and PDF formats."""
    st.markdown("### 📥 Segmentation & RFM Export Portal")
    st.markdown("Download current filtered segmentation results in CSV, Excel, or formatted PDF report formats.")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_bytes, csv_fn, csv_mime = segmentation_service.generate_export_file("csv", filtered_fs_df)
        st.download_button(
            label="📄 Download CSV Segmentation Data",
            data=csv_bytes,
            file_name=csv_fn,
            mime=csv_mime,
            use_container_width=True,
            key="btn_seg_export_csv"
        )

    with c2:
        excel_bytes, excel_fn, excel_mime = segmentation_service.generate_export_file("excel", filtered_fs_df)
        st.download_button(
            label="📊 Download Excel Workbook",
            data=excel_bytes,
            file_name=excel_fn,
            mime=excel_mime,
            use_container_width=True,
            key="btn_seg_export_excel"
        )

    with c3:
        pdf_bytes, pdf_fn, pdf_mime = segmentation_service.generate_export_file("pdf", filtered_fs_df, kpis=kpis)
        st.download_button(
            label="📕 Download PDF Segmentation Report",
            data=pdf_bytes,
            file_name=pdf_fn,
            mime=pdf_mime,
            use_container_width=True,
            key="btn_seg_export_pdf"
        )
