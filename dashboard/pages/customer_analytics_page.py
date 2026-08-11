"""
Customer Analytics Page UI Layout for ECIP Phase 12.
Fully integrated with CustomerAnalyticsService backend.
Provides 10 KPIs, Customer Overview, Demographics (Maps & Treemaps), Purchase Behavior,
Loyalty Tiers Analysis, Revenue Contribution & Pareto (80/20) Rule Analysis,
Recency Activity Rosters, Customer Search Profile Lookup, Business Insights, and Multi-Format Exports.
"""

import streamlit as st
import pandas as pd
from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb
from dashboard.components.kpi_cards import render_glass_kpi_card
from dashboard.components.filters import render_customer_analytics_filter_panel, render_customer_search_input
from dashboard.components.charts import (
    create_line_chart,
    create_customer_growth_chart,
    create_treemap_chart,
    create_state_map_chart,
    create_donut_chart,
    create_bar_chart,
    create_histogram_chart,
    create_pareto_chart,
    create_box_plot
)
from backend.services.customer_analytics_service import customer_analytics_service
from utils.logger import setup_logger

logger = setup_logger("ECIP.CustomerAnalyticsPage")

def render_customer_analytics_layout():
    """Renders complete Customer Analytics UI Page connected to backend services."""
    render_top_header("Customer Intelligence & Analytics")
    render_breadcrumb(["Home", "Analytics", "Customer Analytics"])

    # 1. Sidebar Filters
    filter_opts = customer_analytics_service.get_filter_options()
    filters = render_customer_analytics_filter_panel(filter_opts)

    # 2. Customer Search Bar
    st.markdown("### 🔍 Customer Profile Lookup")
    search_query = render_customer_search_input()

    if search_query.strip():
        search_res = customer_analytics_service.search_customer_profile(search_query)
        _render_customer_search_results(search_res)
        st.markdown("---")

    # 3. Load Backend Payload
    with st.spinner("Compiling enterprise customer intelligence metrics & behavior patterns..."):
        payload = customer_analytics_service.get_customer_analytics_payload(
            date_range=filters.get("date_range"),
            states=filters.get("states"),
            cities=filters.get("cities"),
            categories=filters.get("categories"),
            payment_methods=filters.get("payment_methods"),
            customer_segments=filters.get("customer_segments"),
            revenue_range=filters.get("revenue_range")
        )

    # 4. Datasets Status Notice
    _render_datasets_status_notice(payload.get("datasets_status", {}))

    # 5. Render 10 KPI Cards Grid (2 rows of 5 cards)
    kpis = payload.get("kpis", {})
    _render_10_kpi_grid_section(kpis)

    st.markdown("---")

    # 6. Customer Overview Section
    st.markdown("### 📈 Customer Base & Growth Overview")
    _render_customer_overview_section(payload)

    st.markdown("---")

    # 7. Customer Demographics Section
    st.markdown("### 🗺️ Geographic & Demographics Analysis")
    _render_demographics_section(payload.get("demographics", {}))

    st.markdown("---")

    # 8. Customer Purchase Behavior Section
    st.markdown("### 🛒 Customer Purchase Behavior")
    _render_behavior_section(payload.get("behavior", {}))

    st.markdown("---")

    # 9. Loyalty Analysis Section
    st.markdown("### 💎 Customer Loyalty & Tenure Analysis")
    _render_loyalty_section(payload.get("loyalty", {}))

    st.markdown("---")

    # 10. Revenue Contribution & Pareto (80/20) Section
    st.markdown("### 📊 Revenue Contribution & Pareto (80/20) Analysis")
    _render_revenue_contribution_section(payload.get("revenue_contribution", {}))

    st.markdown("---")

    # 11. Customer Activity & Recency Section
    st.markdown("### 🕒 Recency & Activity Roster")
    _render_activity_section(payload.get("activity", {}))

    st.markdown("---")

    # 12. Automated Business Insights
    st.markdown("### 💡 Customer Intelligence Insights")
    _render_insights_section(payload.get("insights", []))

    st.markdown("---")

    # 13. Reports Export Portal
    _render_export_portal(
        filtered_master_df=payload.get("filtered_master_df", pd.DataFrame()),
        kpis=kpis
    )


def _render_10_kpi_grid_section(kpis: dict):
    """Renders 10 Customer KPI cards in 2 rows of 5 columns."""
    st.markdown("### 🌐 Customer Performance Indicators (KPIs)")

    r1_keys = [
        "total_customers", "active_customers", "returning_customers", "new_customers", "repeat_purchase_rate"
    ]
    r2_keys = [
        "avg_customer_clv", "avg_customer_rating", "customer_retention_rate", "avg_purchase_frequency", "avg_basket_size"
    ]

    # Row 1 (5 Cards)
    cols1 = st.columns(5)
    for idx, k in enumerate(r1_keys):
        m = kpis.get(k, {})
        with cols1[idx]:
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

    # Row 2 (5 Cards)
    cols2 = st.columns(5)
    for idx, k in enumerate(r2_keys):
        m = kpis.get(k, {})
        with cols2[idx]:
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


def _render_customer_overview_section(payload: dict):
    """Renders Customer Growth and Active vs Inactive Overview."""
    charts_data = payload.get("loyalty", {})
    loyalty_trend_df = charts_data.get("loyalty_trend", pd.DataFrame())

    c1, c2 = st.columns([1.6, 1.2])

    with c1:
        fig_trend = create_line_chart(loyalty_trend_df, x="Month", y="Loyal_Buyers_Count", title="Monthly Active Buyers Growth Trajectory")
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        kpis = payload.get("kpis", {})
        active_count = kpis.get("active_customers", {}).get("raw_value", 0)
        tot_count = kpis.get("total_customers", {}).get("raw_value", 1)
        inactive_count = max(0, tot_count - active_count)

        df_act = pd.DataFrame({
            "Status": ["Active Buyers (<=90d)", "Inactive Buyers (>90d)"],
            "Count": [active_count, inactive_count]
        })
        fig_act = create_donut_chart(df_act, names="Status", values="Count", title="Active vs Inactive Customer Ratio")
        st.plotly_chart(fig_act, use_container_width=True)


def _render_demographics_section(demographics: dict):
    """Renders Demographics state map, top cities bar chart, and geo treemap."""
    state_df = demographics.get("state_distribution", pd.DataFrame())
    city_df = demographics.get("city_distribution", pd.DataFrame())
    geo_tree_df = demographics.get("geo_treemap", pd.DataFrame())

    c1, c2 = st.columns([1.4, 1.4])
    with c1:
        fig_map = create_state_map_chart(state_df, title="Customer Concentration by State (Map)")
        st.plotly_chart(fig_map, use_container_width=True)
    with c2:
        fig_city = create_bar_chart(city_df, x="Customer_Count", y="City", horizontal=True, title="Top 15 Cities by Customer Base")
        st.plotly_chart(fig_city, use_container_width=True)

    if not geo_tree_df.empty:
        st.markdown("#### Geographic Revenue Contribution (State → City Hierarchy)")
        fig_geo_tree = create_treemap_chart(geo_tree_df.head(50), path_col="City", values_col="Total_Revenue", title="Geographic Sales Volume Treemap")
        st.plotly_chart(fig_geo_tree, use_container_width=True)


def _render_behavior_section(behavior: dict):
    """Renders purchase frequency distribution, product diversity, payment methods, revenue tiers."""
    freq_df = behavior.get("frequency_distribution", pd.DataFrame())
    div_df = behavior.get("product_diversity", pd.DataFrame())
    pmt_df = behavior.get("payment_methods", pd.DataFrame())
    rev_tier_df = behavior.get("revenue_tiers", pd.DataFrame())

    c1, c2 = st.columns([1.4, 1.4])
    with c1:
        fig_freq = create_bar_chart(freq_df, x="Frequency_Bucket", y="Customer_Count", title="Order Frequency Distribution (Orders per Customer)")
        st.plotly_chart(fig_freq, use_container_width=True)
    with c2:
        fig_div = create_donut_chart(div_df, names="Diversity_Level", values="Customer_Count", title="Product Diversity (Categories Purchased)")
        st.plotly_chart(fig_div, use_container_width=True)

    c3, c4 = st.columns([1.4, 1.4])
    with c3:
        fig_pmt = create_donut_chart(pmt_df, names="Payment_Method", values="Total_Revenue", title="Preferred Payment Methods Breakdown")
        st.plotly_chart(fig_pmt, use_container_width=True)
    with c4:
        fig_tier = create_bar_chart(rev_tier_df, x="Revenue_Tier", y="Customer_Count", title="Customer Revenue Spending Tiers")
        st.plotly_chart(fig_tier, use_container_width=True)


def _render_loyalty_section(loyalty: dict):
    """Renders Loyalty Tiers, Loyalty Score Histogram, and Loyalty Trend."""
    loyalty_tiers_df = loyalty.get("loyalty_tiers", pd.DataFrame())
    score_hist_df = loyalty.get("loyalty_score_histogram", pd.DataFrame())

    c1, c2 = st.columns([1.4, 1.4])
    with c1:
        fig_loyalty = create_donut_chart(loyalty_tiers_df, names="Loyalty_Tier", values="Customer_Count", title="Customer Loyalty Tier Stratification")
        st.plotly_chart(fig_loyalty, use_container_width=True)
    with c2:
        fig_hist = create_histogram_chart(score_hist_df, x="Score_Range", y="Customer_Count", title="Loyalty Score Index Histogram (0-100)")
        st.plotly_chart(fig_hist, use_container_width=True)


def _render_revenue_contribution_section(revenue_contrib: dict):
    """Renders Pareto 80/20 Chart and Top 20 Customers Roster."""
    pareto_res = revenue_contrib.get("pareto", {})
    pareto_df = pareto_res.get("pareto_df", pd.DataFrame())
    summary_stat = pareto_res.get("summary_stat", "")

    top_20_df = revenue_contrib.get("top_20_customers", pd.DataFrame())
    quantiles_df = revenue_contrib.get("revenue_quantiles", pd.DataFrame())

    st.markdown(summary_stat)

    c1, c2 = st.columns([1.5, 1.3])
    with c1:
        fig_pareto = create_pareto_chart(pareto_df, x="Customer_Percentile", y="Cumulative_Revenue_Pct", title="Pareto 80/20 Revenue Concentration Curve")
        st.plotly_chart(fig_pareto, use_container_width=True)
    with c2:
        if not quantiles_df.empty:
            st.markdown("#### Customer Revenue Quantile Distribution")
            fig_q = create_bar_chart(quantiles_df, x="Revenue_Share_Pct", y="Segment", horizontal=True, title="Revenue Share by Quantile Tier (%)")
            st.plotly_chart(fig_q, use_container_width=True)

    if not top_20_df.empty:
        with st.expander("🏆 Top 20 Customers by Revenue Leaderboard", expanded=True):
            st.dataframe(top_20_df, use_container_width=True)


def _render_activity_section(activity: dict):
    """Renders Recency Distribution, Recently Active Customers, and Dormant Customers."""
    rec_dist_df = activity.get("recency_distribution", pd.DataFrame())
    recent_active_df = activity.get("recently_active", pd.DataFrame())
    dormant_df = activity.get("dormant_customers", pd.DataFrame())

    fig_rec = create_bar_chart(rec_dist_df, x="Recency_Bucket", y="Customer_Count", title="Customer Inactivity Recency Distribution")
    st.plotly_chart(fig_rec, use_container_width=True)

    tab1, tab2 = st.tabs(["🟢 Recently Active Roster (Top 50)", "🔴 Dormant Customers Roster (>90 Days Inactive)"])
    with tab1:
        if not recent_active_df.empty:
            st.dataframe(recent_active_df, use_container_width=True)
        else:
            st.info("No recently active customers found.")
    with tab2:
        if not dormant_df.empty:
            st.dataframe(dormant_df, use_container_width=True)
        else:
            st.info("No dormant customer records found.")


def _render_customer_search_results(result: dict):
    """Renders structured Customer Search Profile card and transaction history."""
    if not result.get("has_match"):
        st.warning(result.get("message", f"No customer profile found for query '{result.get('query')}'"))
        return

    m_type = result.get("match_type", "Customer")
    profile = result.get("profile", {})
    records = result.get("records", pd.DataFrame())

    st.success(f"🎯 **Found Matching {m_type}**: {result.get('query')}")

    with st.expander(f"👤 {m_type} Detailed Profile Card", expanded=True):
        p_cols = st.columns(min(4, len(profile)))
        for idx, (k, v) in enumerate(profile.items()):
            col_idx = idx % min(4, len(profile))
            with p_cols[col_idx]:
                st.metric(label=k, value=str(v))

        if not records.empty:
            st.markdown("#### Transaction History")
            st.dataframe(records.head(30), use_container_width=True)


def _render_insights_section(insights: list):
    """Renders 6 Customer Intelligence insight cards."""
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


def _render_datasets_status_notice(status: dict):
    """Notice banner for dataset availability."""
    missing = [k for k, v in status.items() if not v.get("available")]
    if missing:
        st.warning(f"⚠️ **Notice**: Pipeline datasets missing: `{', '.join(missing)}`. Fallback datasets or raw merged files are active.")


def _render_export_portal(filtered_master_df: pd.DataFrame, kpis: dict):
    """Renders export buttons for CSV, Excel, and PDF formats."""
    st.markdown("### 📥 Customer Analytics Export Portal")
    st.markdown("Download current filtered customer analytics dataset in CSV, Excel, or formatted PDF report formats.")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_bytes, csv_fn, csv_mime = customer_analytics_service.generate_export_file("csv", filtered_master_df)
        st.download_button(
            label="📄 Download CSV Customer Data",
            data=csv_bytes,
            file_name=csv_fn,
            mime=csv_mime,
            use_container_width=True,
            key="btn_cust_export_csv"
        )

    with c2:
        excel_bytes, excel_fn, excel_mime = customer_analytics_service.generate_export_file("excel", filtered_master_df)
        st.download_button(
            label="📊 Download Excel Customer Workbook",
            data=excel_bytes,
            file_name=excel_fn,
            mime=excel_mime,
            use_container_width=True,
            key="btn_cust_export_excel"
        )

    with c3:
        pdf_bytes, pdf_fn, pdf_mime = customer_analytics_service.generate_export_file("pdf", filtered_master_df, kpis=kpis)
        st.download_button(
            label="📕 Download Customer PDF Report",
            data=pdf_bytes,
            file_name=pdf_fn,
            mime=pdf_mime,
            use_container_width=True,
            key="btn_cust_export_pdf"
        )
