"""
Customer Lifetime Value (CLV) & Revenue Intelligence Page UI Layout for ECIP Phase 15.
Fully integrated with CLVService backend.
Provides 8 KPIs, CLV Overview, Customer Value Explorer, 5-Tier Value Stratification,
Revenue Intelligence (Top 100 Leaderboard & Pareto Concentration), Opportunity Intelligence,
SHAP Explainable AI for CLV Regression, Multi-Horizon Revenue Forecasting, Business Insights,
CLV Search, and Multi-Format Exports.
"""

import streamlit as st
import pandas as pd
from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb
from dashboard.components.kpi_cards import render_glass_kpi_card
from dashboard.components.filters import render_clv_filter_panel, render_clv_search_input
from dashboard.components.charts import (
    create_line_chart,
    create_treemap_chart,
    create_donut_chart,
    create_bar_chart,
    create_histogram_chart,
    create_pareto_chart
)
from backend.services.clv_service import clv_service
from utils.logger import setup_logger

logger = setup_logger("ECIP.CLVPage")

def render_clv_layout():
    """Renders complete Customer Lifetime Value (CLV) & Revenue Intelligence Dashboard Page."""
    render_top_header("Customer Lifetime Value (CLV) & Revenue Intelligence")
    render_breadcrumb(["Home", "Analytics", "Customer Lifetime Value"])

    # 1. Sidebar Filters
    filter_opts = clv_service.get_filter_options()
    filters = render_clv_filter_panel(filter_opts)

    # 2. CLV Search Bar
    st.markdown("### 🔍 CLV & Revenue Intelligence Search")
    search_query = render_clv_search_input()

    if search_query.strip():
        search_res = clv_service.search_clv_profile(search_query)
        _render_clv_search_results(search_res)
        st.markdown("---")

    # 3. Load Backend Payload
    with st.spinner("Compiling ML lifetime value predictions, 12-month revenue forecasts, and SHAP attributions..."):
        payload = clv_service.get_clv_payload(
            date_range=filters.get("date_range"),
            tiers=filters.get("tiers"),
            clusters=filters.get("clusters"),
            states=filters.get("states"),
            categories=filters.get("categories"),
            revenue_range=filters.get("revenue_range"),
            clv_range=filters.get("clv_range"),
            forecast_period=filters.get("forecast_period", "Monthly")
        )

    # 4. Datasets Status Notice
    _render_datasets_status_notice(payload.get("datasets_status", {}))

    # 5. Render 8 KPI Cards Grid
    kpis = payload.get("kpis", {})
    _render_8_kpi_grid_section(kpis)

    st.markdown("---")

    # 6. CLV Overview Section
    st.markdown("### 🌐 CLV Distribution & Revenue Forecast Overview")
    _render_clv_overview_section(payload.get("value_tier_matrix", pd.DataFrame()), payload.get("forecast", {}))

    st.markdown("---")

    # 7. Customer Value Explorer
    st.markdown("### 🔍 Customer Value Explorer")
    _render_customer_value_explorer(payload.get("filtered_clv_df", pd.DataFrame()))

    st.markdown("---")

    # 8. 5-Tier Customer Value Classification
    st.markdown("### 🏆 5-Tier Customer Value Stratification")
    _render_value_classification_section(payload.get("value_tier_matrix", pd.DataFrame()))

    st.markdown("---")

    # 9. Revenue Intelligence & Pareto Concentration
    st.markdown("### 📊 Revenue Intelligence & Concentration")
    _render_revenue_intelligence_section(payload.get("top_100_leaderboard", pd.DataFrame()), payload.get("pareto_curve", pd.DataFrame()))

    st.markdown("---")

    # 10. Opportunity Intelligence Recommendations
    st.markdown("### 💡 Revenue Growth Opportunity Intelligence")
    _render_opportunity_intelligence_section(payload.get("opportunity_recommendations", []))

    st.markdown("---")

    # 11. Explainable AI (SHAP for CLV)
    st.markdown("### 🧬 Explainable AI (SHAP) for Lifetime Value Predictions")
    _render_explainable_ai_section(payload.get("global_shap", pd.DataFrame()), payload.get("filtered_clv_df", pd.DataFrame()))

    st.markdown("---")

    # 12. Multi-Horizon Revenue Forecasting
    st.markdown("### 📈 Multi-Horizon Revenue Forecast Comparison")
    _render_revenue_forecasting_section(payload.get("forecast", {}))

    st.markdown("---")

    # 13. Executive Business Insights
    st.markdown("### 💡 Executive Revenue Insights")
    _render_business_insights_section(payload.get("business_insights", []))

    st.markdown("---")

    # 14. Reports Export Portal
    _render_export_portal(
        filtered_clv_df=payload.get("filtered_clv_df", pd.DataFrame()),
        kpis=kpis
    )


def _render_8_kpi_grid_section(kpis: dict):
    """Renders 8 CLV KPI cards in 2 rows of 4 columns."""
    st.markdown("### 🌐 Lifetime Value Performance Indicators (KPIs)")

    kpi_keys = [
        "total_predicted_clv", "avg_customer_clv", "highest_value_customer", "high_value_customers",
        "platinum_customers", "expected_revenue_12m", "avg_revenue_per_customer", "revenue_growth_potential"
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
                icon=m.get("icon", "💎"),
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
                icon=m.get("icon", "💎"),
                badge_type="green" if m.get("is_positive", True) else "red",
                last_updated=m.get("last_updated", "")
            )


def _render_clv_overview_section(tier_df: pd.DataFrame, forecast: dict):
    """Renders Revenue Share by Tier Treemap & Revenue Forecast Line Chart."""
    c1, c2 = st.columns([1.4, 1.4])

    with c1:
        if not tier_df.empty:
            fig_tree = create_treemap_chart(tier_df, path_col="Value_Tier", values_col="Total_Revenue", title="Revenue Contribution Share by Tier ($)")
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No tier distribution data available.")

    with c2:
        fdata = forecast.get("forecast_data", pd.DataFrame())
        if not fdata.empty and "Period" in fdata.columns:
            fig_line = create_line_chart(fdata, x="Period", y="Predicted_Revenue", title="Monthly Revenue Forecast Trend ($)")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No forecast trend data available.")


def _render_customer_value_explorer(clv_df: pd.DataFrame):
    """Renders interactive dropdown selector to inspect deep-dive CLV customer profiles."""
    if clv_df.empty:
        st.info("No customer CLV records available.")
        return

    clv_col = "predicted_clv" if "predicted_clv" in clv_df.columns else ("historical_clv" if "historical_clv" in clv_df.columns else "total_spending")
    cust_col = "customer_unique_id" if "customer_unique_id" in clv_df.columns else clv_df.columns[0]

    sorted_df = clv_df.sort_values(by=clv_col, ascending=False)
    cust_list = sorted_df[cust_col].astype(str).tolist()

    selected_cust = st.selectbox(
        "Select Customer to Inspect Value Profile",
        options=cust_list,
        index=0,
        key="select_clv_cust_dropdown"
    )

    row = sorted_df[sorted_df[cust_col].astype(str) == selected_cust].iloc[0]

    clv_val = float(row.get(clv_col, 250.0))
    tier = clv_service.value_classifier.classify_customer_tier(clv_val)
    spending = float(row.get("total_spending", clv_val))
    orders = int(row.get("total_orders", 1))
    recency = int(row.get("recency_days", 45))
    loyalty = float(row.get("loyalty_score", 65.0))
    csat = float(row.get("avg_review_score_given", 4.2))

    st.markdown(f"#### 📌 Value Profile: **{selected_cust}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted 12m CLV", f"${clv_val:,.2f}")
        st.metric("Customer Tier", tier)
    with c2:
        st.metric("Total Spending", f"${spending:,.2f}")
        st.metric("Total Orders", f"{orders:,}")
    with c3:
        st.metric("Loyalty Score Index", f"{loyalty:.1f} / 100")
        st.metric("Average CSAT Rating", f"{csat:.1f} / 5.0")
    with c4:
        st.metric("Recency Inactivity", f"{recency} Days")
        st.metric("Preferred Payment", "Credit Card")


def _render_value_classification_section(tier_df: pd.DataFrame):
    """Renders 5 Customer Value Tiers (Platinum to Standard) summary matrix."""
    if tier_df.empty:
        st.info("No value tier classification data available.")
        return

    c1, c2 = st.columns([1.5, 1.3])
    with c1:
        fig_bar = create_bar_chart(tier_df, x="Value_Tier", y="Customer_Count", title="Accounts Volume by Customer Value Tier")
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        st.markdown("#### 📋 5-Tier Customer Stratification Matrix")
        display_df = tier_df.copy()
        display_df["Total_Revenue"] = display_df["Total_Revenue"].apply(lambda v: f"${v:,.2f}")
        display_df["Avg_Spending"] = display_df["Avg_Spending"].apply(lambda v: f"${v:,.2f}")
        display_df["Avg_Orders"] = display_df["Avg_Orders"].round(1)
        st.dataframe(display_df, use_container_width=True)


def _render_revenue_intelligence_section(top_100_df: pd.DataFrame, pareto_df: pd.DataFrame):
    """Renders Top 100 Leaderboard and Pareto 80/20 Concentration curve."""
    tab1, tab2 = st.tabs(["👑 Top 100 High-Value Leaderboard", "📈 Pareto (80/20) Revenue Concentration"])

    with tab1:
        if not top_100_df.empty:
            st.dataframe(top_100_df, use_container_width=True)
        else:
            st.info("No leaderboard data available.")

    with tab2:
        if not pareto_df.empty:
            fig_pareto = create_pareto_chart(pareto_df, x="Customer_Percentile", y="Cumulative_Revenue_Pct", title="Pareto 80/20 Cumulative Revenue Concentration (%)")
            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.info("No Pareto curve data available.")


def _render_opportunity_intelligence_section(opportunities: list):
    """Renders 6 revenue growth opportunity intelligence cards."""
    if not opportunities:
        return

    cols = st.columns(min(3, len(opportunities)))
    for idx, opp in enumerate(opportunities):
        col_idx = idx % min(3, len(opportunities))
        with cols[col_idx]:
            badge = opp.get("badge_color", "purple")
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 14px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div style="font-size: 20px;">{opp.get('icon', '💡')}</div>
                        <span class="badge-{badge}">{opp.get('priority')}</span>
                    </div>
                    <div style="font-weight: 700; font-size: 15px; color: #f8fafc; margin-top: 6px;">{opp.get('title')}</div>
                    <div style="font-size: 12px; color: #38bdf8; margin: 4px 0;">{opp.get('target_group')} ({opp.get('candidate_count')})</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px;"><b>Reason:</b> {opp.get('business_reason')}</div>
                    <div style="font-size: 12px; color: #34d399; margin-top: 6px;">
                        <b>Revenue Impact:</b> {opp.get('estimated_revenue_impact')}<br>
                        <b>Confidence:</b> {opp.get('confidence_score')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_explainable_ai_section(global_shap_df: pd.DataFrame, clv_df: pd.DataFrame):
    """Renders Global SHAP Feature Importance and Local Diagnostic Explanation."""
    tab1, tab2 = st.tabs(["📊 Global CLV Model Drivers", "🔍 Individual Customer Attributions"])

    with tab1:
        if not global_shap_df.empty:
            fig_shap = create_bar_chart(global_shap_df, x="Importance_Weight", y="Feature", horizontal=True, title="Global Feature Drivers for Lifetime Value")
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("No global SHAP feature data available.")

    with tab2:
        if not clv_df.empty:
            cust_col = "customer_unique_id" if "customer_unique_id" in clv_df.columns else clv_df.columns[0]
            cid = str(clv_df[cust_col].iloc[0])
            explanation = clv_service.explainability_engine.explain_customer_clv(cid, clv_df)

            st.markdown(f"#### Plain-English Narrative: **{cid}**")
            st.info(explanation.get("plain_english_summary"))

            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Top Positive Value Drivers (+$)**")
                for f in explanation.get("top_positive_drivers", []):
                    st.markdown(f"• `{f}`")
            with p2:
                st.markdown("**Top Negative / Drag Factors (-$)**")
                for f in explanation.get("top_negative_drivers", []):
                    st.markdown(f"• `{f}`")


def _render_revenue_forecasting_section(forecast: dict):
    """Renders Multi-Horizon Revenue Forecast table & Actual vs Predicted chart."""
    fdata = forecast.get("forecast_data", pd.DataFrame())
    period = forecast.get("period_type", "Monthly")

    if fdata.empty:
        st.info("No revenue forecast dataset available.")
        return

    c1, c2 = st.columns([1.5, 1.3])
    with c1:
        fig_fore = create_line_chart(fdata, x="Period", y="Predicted_Revenue", title=f"{period} Revenue Forecast ($)")
        st.plotly_chart(fig_fore, use_container_width=True)
    with c2:
        st.markdown(f"#### 📋 {period} Forecast Data Table")
        disp_df = fdata.copy()
        disp_df["Actual_Revenue"] = disp_df["Actual_Revenue"].apply(lambda v: f"${v:,.2f}" if not pd.isna(v) else "Projected")
        disp_df["Predicted_Revenue"] = disp_df["Predicted_Revenue"].apply(lambda v: f"${v:,.2f}" if not pd.isna(v) else "N/A")
        st.dataframe(disp_df, use_container_width=True)


def _render_business_insights_section(insights: list):
    """Renders 6 automated executive revenue insight cards."""
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
                    <div style="font-size: 13px; color: #38bdf8; font-weight: 700; margin: 2px 0;">{ins.get('metric')}</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">{ins.get('detail')}</div>
                    <div style="font-size: 11px; color: #34d399; margin-top: 4px;"><b>Recommendation:</b> {ins.get('recommendation')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_clv_search_results(result: dict):
    """Renders CLV Search Profile card."""
    if not result.get("has_match"):
        st.warning(result.get("message", f"No CLV result for query '{result.get('query')}'"))
        return

    m_type = result.get("match_type", "Entity")
    profile = result.get("profile", {})
    explanation = result.get("explanation", {})

    st.success(f"🎯 **Found Matching {m_type}**: {result.get('query')}")

    with st.expander(f"📌 {m_type} Detailed CLV Profile", expanded=True):
        p_cols = st.columns(min(4, len(profile)))
        for idx, (k, v) in enumerate(profile.items()):
            col_idx = idx % min(4, len(profile))
            with p_cols[col_idx]:
                st.metric(label=k, value=str(v))

        if explanation:
            st.info(explanation.get("plain_english_summary", ""))


def _render_datasets_status_notice(status: dict):
    """Notice banner for dataset availability."""
    missing = [k for k, v in status.items() if not v.get("available")]
    if missing:
        st.warning(f"⚠️ **Notice**: Pipeline datasets missing: `{', '.join(missing)}`. Fallback predictions active.")


def _render_export_portal(filtered_clv_df: pd.DataFrame, kpis: dict):
    """Renders export buttons for CSV, Excel, and PDF formats."""
    st.markdown("### 📥 CLV & Revenue Export Portal")
    st.markdown("Download current filtered CLV predictions and revenue analysis in CSV, Excel, or formatted PDF report formats.")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_bytes, csv_fn, csv_mime = clv_service.generate_export_file("csv", filtered_clv_df)
        st.download_button(
            label="📄 Download CSV CLV Predictions",
            data=csv_bytes,
            file_name=csv_fn,
            mime=csv_mime,
            use_container_width=True,
            key="btn_clv_export_csv"
        )

    with c2:
        excel_bytes, excel_fn, excel_mime = clv_service.generate_export_file("excel", filtered_clv_df)
        st.download_button(
            label="📊 Download Excel Workbook",
            data=excel_bytes,
            file_name=excel_fn,
            mime=excel_mime,
            use_container_width=True,
            key="btn_clv_export_excel"
        )

    with c3:
        pdf_bytes, pdf_fn, pdf_mime = clv_service.generate_export_file("pdf", filtered_clv_df, kpis=kpis)
        st.download_button(
            label="📕 Download PDF Revenue Report",
            data=pdf_bytes,
            file_name=pdf_fn,
            mime=pdf_mime,
            use_container_width=True,
            key="btn_clv_export_pdf"
        )
