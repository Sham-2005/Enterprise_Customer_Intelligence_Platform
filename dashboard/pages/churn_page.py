"""
AI Customer Churn Prediction & Risk Intelligence Page UI Layout for ECIP Phase 14.
Fully integrated with ChurnService backend.
Provides 8 KPIs, Churn Overview, High-Risk Explorer, SHAP Explainable AI (XAI),
5-Tier Risk Stratification, Personalized Retention Intelligence, Customer Timelines,
Batch CSV Prediction Portal, Business Insights, Churn Search, and Multi-Format Exports.
"""

import streamlit as st
import pandas as pd
from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb
from dashboard.components.kpi_cards import render_glass_kpi_card
from dashboard.components.filters import render_churn_filter_panel, render_churn_search_input
from dashboard.components.charts import (
    create_line_chart,
    create_treemap_chart,
    create_donut_chart,
    create_bar_chart,
    create_histogram_chart
)
from backend.services.churn_service import churn_service
from utils.logger import setup_logger

logger = setup_logger("ECIP.ChurnPage")

def render_churn_layout():
    """Renders complete AI Customer Churn Prediction & Risk Intelligence Dashboard Page."""
    render_top_header("AI Customer Churn & Risk Intelligence Dashboard")
    render_breadcrumb(["Home", "AI Models", "Customer Churn Prediction"])

    # 1. Sidebar Filters
    filter_opts = churn_service.get_filter_options()
    filters = render_churn_filter_panel(filter_opts)

    # 2. Churn Search Bar
    st.markdown("### 🔍 Churn Risk Intelligence Search")
    search_query = render_churn_search_input()

    if search_query.strip():
        search_res = churn_service.search_churn_profile(search_query)
        _render_churn_search_results(search_res)
        st.markdown("---")

    # 3. Load Backend Payload
    with st.spinner("Executing AI churn models, SHAP feature attributions, and risk stratification..."):
        payload = churn_service.get_churn_payload(
            date_range=filters.get("date_range"),
            risk_levels=filters.get("risk_levels"),
            clusters=filters.get("clusters"),
            states=filters.get("states"),
            categories=filters.get("categories"),
            revenue_range=filters.get("revenue_range"),
            clv_range=filters.get("clv_range")
        )

    # 4. Datasets Status Notice
    _render_datasets_status_notice(payload.get("datasets_status", {}))

    # 5. Render 8 KPI Cards Grid
    kpis = payload.get("kpis", {})
    _render_8_kpi_grid_section(kpis)

    st.markdown("---")

    # 6. Churn Overview Section
    st.markdown("### 🌐 Churn Overview & Revenue at Risk")
    _render_churn_overview_section(payload.get("filtered_churn_df", pd.DataFrame()), payload.get("risk_distribution", pd.DataFrame()))

    st.markdown("---")

    # 7. High-Risk Customer Explorer
    st.markdown("### 🔍 High-Risk Customer Explorer")
    _render_high_risk_explorer(payload.get("filtered_churn_df", pd.DataFrame()))

    st.markdown("---")

    # 8. Explainable AI (XAI) SHAP Section
    st.markdown("### 🧬 Explainable AI (XAI) SHAP Feature Attributions")
    _render_explainable_ai_section(payload.get("global_shap", pd.DataFrame()), payload.get("filtered_churn_df", pd.DataFrame()))

    st.markdown("---")

    # 9. 5-Tier Risk Stratification
    st.markdown("### ⚖️ 5-Tier Risk Stratification")
    _render_risk_classification_section(payload.get("risk_distribution", pd.DataFrame()))

    st.markdown("---")

    # 10. Retention Intelligence Campaigns
    st.markdown("### 🛡️ Personalized Retention Campaigns")
    _render_retention_intelligence_section(payload.get("retention_recommendations", []))

    st.markdown("---")

    # 11. Customer Timeline Section
    st.markdown("### ⏳ Customer Purchase History & Risk Trajectory")
    _render_customer_timeline_section(payload.get("filtered_master_df", pd.DataFrame()), payload.get("filtered_churn_df", pd.DataFrame()))

    st.markdown("---")

    # 12. Batch CSV Prediction Portal
    st.markdown("### 📤 Batch CSV Churn Prediction Engine")
    _render_batch_prediction_portal()

    st.markdown("---")

    # 13. Business Insights Cards
    st.markdown("### 💡 Strategic Business Insights")
    _render_business_insights_section(payload.get("business_insights", []))

    st.markdown("---")

    # 14. Reports Export Portal
    _render_export_portal(
        filtered_churn_df=payload.get("filtered_churn_df", pd.DataFrame()),
        kpis=kpis
    )


def _render_8_kpi_grid_section(kpis: dict):
    """Renders 8 Churn KPI cards in 2 rows of 4 columns."""
    st.markdown("### 🌐 Churn & Risk Performance Indicators (KPIs)")

    kpi_keys = [
        "total_customers", "high_risk_customers", "critical_risk_customers", "avg_churn_probability",
        "predicted_churn_rate", "retention_success_estimate", "avg_customer_clv", "estimated_revenue_at_risk"
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
                icon=m.get("icon", "📊"),
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
                icon=m.get("icon", "📊"),
                badge_type="green" if m.get("is_positive", True) else "red",
                last_updated=m.get("last_updated", "")
            )


def _render_churn_overview_section(churn_df: pd.DataFrame, risk_df: pd.DataFrame):
    """Renders Churn Donut chart and Revenue at Risk by Tier Treemap/Bar."""
    c1, c2 = st.columns([1.4, 1.4])

    with c1:
        if not risk_df.empty:
            fig_donut = create_donut_chart(risk_df, names="Risk_Tier", values="Customer_Count", title="Risk Level Distribution (% Accounts)")
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No risk distribution data available.")

    with c2:
        if not risk_df.empty:
            fig_rev = create_treemap_chart(risk_df, path_col="Risk_Tier", values_col="Total_Revenue", title="Revenue at Risk by Tier ($)")
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No revenue at risk data available.")


def _render_high_risk_explorer(churn_df: pd.DataFrame):
    """Renders interactive customer selector for High and Critical risk accounts."""
    if churn_df.empty:
        st.info("No active customer prediction records available.")
        return

    cust_col = "customer_unique_id" if "customer_unique_id" in churn_df.columns else churn_df.columns[0]
    if "churn_probability" in churn_df.columns:
        high_risk_df = churn_df[churn_df["churn_probability"] >= 0.5]
    else:
        high_risk_df = churn_df

    if high_risk_df.empty:
        high_risk_df = churn_df

    cust_list = high_risk_df[cust_col].astype(str).tolist()

    selected_cust = st.selectbox(
        "Select At-Risk Customer to Inspect Profile",
        options=cust_list,
        index=0,
        key="select_high_risk_cust_dropdown"
    )

    row = high_risk_df[high_risk_df[cust_col].astype(str) == selected_cust].iloc[0]

    prob = float(row.get("churn_probability", 0.75))
    risk = row.get("risk_level", "High Risk")
    spending = float(row.get("total_spending", 250.0))
    orders = int(row.get("total_orders", 1))
    recency = int(row.get("recency_days", 90))
    csat = float(row.get("avg_review_score_given", 3.0))

    st.markdown(f"#### 📌 Deep-Dive Profile: **{selected_cust}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted Churn Prob", f"{prob * 100:.1f}%")
        st.metric("Risk Classification", risk)
    with c2:
        st.metric("Total Spending", f"${spending:,.2f}")
        st.metric("Total Orders", f"{orders:,}")
    with c3:
        st.metric("Recency Inactivity", f"{recency} Days")
        st.metric("Average CSAT Rating", f"{csat:.1f} / 5.0")
    with c4:
        st.metric("Customer Segment", row.get("rfm_segment", row.get("cluster_name", "Champions")))
        st.metric("Preferred Payment", "Credit Card")


def _render_explainable_ai_section(global_shap_df: pd.DataFrame, churn_df: pd.DataFrame):
    """Renders Global SHAP Importance & Local Diagnostic Narrative."""
    tab1, tab2 = st.tabs(["📊 Global SHAP Drivers", "🔍 Local Customer Explanation"])

    with tab1:
        if not global_shap_df.empty:
            fig_shap = create_bar_chart(global_shap_df, x="Importance_Weight", y="Feature", horizontal=True, title="Global Feature Drivers for Customer Churn")
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("No global SHAP feature data available.")

    with tab2:
        if not churn_df.empty:
            cust_col = "customer_unique_id" if "customer_unique_id" in churn_df.columns else churn_df.columns[0]
            cid = str(churn_df[cust_col].iloc[0])
            explanation = churn_service.explainability_engine.explain_customer(cid, churn_df)

            st.markdown(f"#### Plain-English Diagnostic: **{cid}**")
            st.info(explanation.get("plain_english_explanation"))

            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Top Positive Risk Drivers (Increasing Churn)**")
                for f in explanation.get("top_positive_risk_factors", []):
                    st.markdown(f"• `{f}`")
            with p2:
                st.markdown("**Top Protective Factors (Reducing Churn)**")
                for f in explanation.get("top_negative_risk_factors", []):
                    st.markdown(f"• `{f}`")


def _render_risk_classification_section(risk_df: pd.DataFrame):
    """Renders 5 Risk Tiers breakdown table and bar chart."""
    if risk_df.empty:
        st.info("No risk classification data available.")
        return

    c1, c2 = st.columns([1.5, 1.3])
    with c1:
        fig_bar = create_bar_chart(risk_df, x="Risk_Tier", y="Customer_Count", title="Accounts Volume by Risk Tier")
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        st.markdown("#### 📋 Risk Tiers Summary Matrix")
        risk_df_display = risk_df.copy()
        risk_df_display["Total_Revenue"] = risk_df_display["Total_Revenue"].apply(lambda v: f"${v:,.2f}")
        st.dataframe(risk_df_display, use_container_width=True)


def _render_retention_intelligence_section(recommendations: list):
    """Renders personalized retention campaign cards."""
    if not recommendations:
        return

    cols = st.columns(min(3, len(recommendations)))
    for idx, rec in enumerate(recommendations):
        col_idx = idx % min(3, len(recommendations))
        with cols[col_idx]:
            badge = rec.get("badge_color", "cyan")
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 14px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div style="font-size: 20px;">{rec.get('icon', '🛡️')}</div>
                        <span class="badge-{badge}">{rec.get('priority')}</span>
                    </div>
                    <div style="font-weight: 700; font-size: 15px; color: #f8fafc; margin-top: 6px;">{rec.get('title')}</div>
                    <div style="font-size: 12px; color: #38bdf8; margin: 4px 0;">{rec.get('target_segment')}</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px;"><b>Action:</b> {rec.get('action_plan')}</div>
                    <div style="font-size: 12px; color: #34d399; margin-top: 6px;">
                        <b>Impact:</b> {rec.get('estimated_impact')} | <b>Saved:</b> {rec.get('expected_revenue_saved')}<br>
                        <b>Confidence:</b> {rec.get('confidence_score')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_customer_timeline_section(master_df: pd.DataFrame, churn_df: pd.DataFrame):
    """Renders purchase history timeline and risk trajectory curve."""
    if churn_df.empty:
        return

    cust_col = "customer_unique_id" if "customer_unique_id" in churn_df.columns else churn_df.columns[0]
    cid = str(churn_df[cust_col].iloc[0])

    timeline_data = churn_service.timeline_engine.get_customer_timeline(cid, master_df, churn_df)
    events_df = timeline_data.get("timeline_events", pd.DataFrame())
    risk_traj_df = timeline_data.get("risk_trajectory", pd.DataFrame())

    st.markdown(f"#### 📅 Timeline & Risk Trajectory for **{cid}**")

    t1, t2 = st.columns([1.5, 1.3])
    with t1:
        if not risk_traj_df.empty and "Date" in risk_traj_df.columns:
            fig_traj = create_line_chart(risk_traj_df, x="Date", y="Churn_Probability_Pct", title="Historical Churn Risk Trajectory (%)")
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("No risk trajectory data available.")
    with t2:
        if not events_df.empty:
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("No transaction history events available.")


def _render_batch_prediction_portal():
    """Renders CSV file uploader for batch churn prediction."""
    uploaded_file = st.file_uploader("Upload Customer Feature Store CSV for Batch Churn Prediction", type=["csv"], key="uploader_batch_churn_csv")

    if uploaded_file is not None:
        with st.spinner("Scoring batch dataset against trained machine learning pipeline..."):
            pred_df, metrics = churn_service.execute_batch_prediction(uploaded_file)

        if not pred_df.empty:
            st.success(f"✅ Batch prediction complete: Scored **{metrics.get('total_records'):,}** records. Found **{metrics.get('high_critical_risk_count'):,}** High/Critical risk accounts.")

            st.dataframe(pred_df.head(50), use_container_width=True)

            csv_bytes = churn_service.export_service.export_to_csv(pred_df)
            st.download_button(
                label="📥 Download Batch Predictions CSV",
                data=csv_bytes,
                file_name="batch_churn_predictions.csv",
                mime="text/csv",
                key="btn_download_batch_csv"
            )


def _render_business_insights_section(insights: list):
    """Renders 5-6 automated business insight cards."""
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
                    <div style="font-size: 13px; color: #f43f5e; font-weight: 700; margin: 2px 0;">{ins.get('metric')}</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">{ins.get('detail')}</div>
                    <div style="font-size: 11px; color: #34d399; margin-top: 4px;"><b>Recommendation:</b> {ins.get('recommendation')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_churn_search_results(result: dict):
    """Renders Churn Search Profile card."""
    if not result.get("has_match"):
        st.warning(result.get("message", f"No churn result for query '{result.get('query')}'"))
        return

    m_type = result.get("match_type", "Entity")
    profile = result.get("profile", {})
    explanation = result.get("explanation", {})

    st.success(f"🎯 **Found Matching {m_type}**: {result.get('query')}")

    with st.expander(f"📌 {m_type} Detailed Churn & SHAP Profile", expanded=True):
        p_cols = st.columns(min(4, len(profile)))
        for idx, (k, v) in enumerate(profile.items()):
            col_idx = idx % min(4, len(profile))
            with p_cols[col_idx]:
                st.metric(label=k, value=str(v))

        st.info(explanation.get("plain_english_explanation", ""))


def _render_datasets_status_notice(status: dict):
    """Notice banner for dataset availability."""
    missing = [k for k, v in status.items() if not v.get("available")]
    if missing:
        st.warning(f"⚠️ **Notice**: Pipeline datasets missing: `{', '.join(missing)}`. Fallback predictions active.")


def _render_export_portal(filtered_churn_df: pd.DataFrame, kpis: dict):
    """Renders export buttons for CSV, Excel, and PDF formats."""
    st.markdown("### 📥 Churn & Risk Predictions Export Portal")
    st.markdown("Download current filtered churn predictions in CSV, Excel, or formatted PDF report formats.")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_bytes, csv_fn, csv_mime = churn_service.generate_export_file("csv", filtered_churn_df)
        st.download_button(
            label="📄 Download CSV Churn Predictions",
            data=csv_bytes,
            file_name=csv_fn,
            mime=csv_mime,
            use_container_width=True,
            key="btn_churn_export_csv"
        )

    with c2:
        excel_bytes, excel_fn, excel_mime = churn_service.generate_export_file("excel", filtered_churn_df)
        st.download_button(
            label="📊 Download Excel Workbook",
            data=excel_bytes,
            file_name=excel_fn,
            mime=excel_mime,
            use_container_width=True,
            key="btn_churn_export_excel"
        )

    with c3:
        pdf_bytes, pdf_fn, pdf_mime = churn_service.generate_export_file("pdf", filtered_churn_df, kpis=kpis)
        st.download_button(
            label="📕 Download PDF Risk Report",
            data=pdf_bytes,
            file_name=pdf_fn,
            mime=pdf_mime,
            use_container_width=True,
            key="btn_churn_export_pdf"
        )
