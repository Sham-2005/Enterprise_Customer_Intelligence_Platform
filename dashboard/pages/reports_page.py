"""
Enterprise Reports & Export Center Page UI Layout for ECIP Phase 19.
Fully integrated with ReportsService backend.
Provides Report Catalog, Category Tabs, Report Preview Engine, Multi-Format PDF/Excel/CSV Generation,
Global Filters, Report History Logging, and Report Storage Persistence.
"""

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb
from dashboard.components.kpi_cards import render_glass_kpi_card, render_kpi_grid_row
from dashboard.components.filters import (
    render_reports_filter_panel,
    render_reports_search_input
)
from backend.services.reports_service import reports_service
from utils.logger import setup_logger

logger = setup_logger("ECIP.ReportsPage")

def render_reports_layout():
    """Renders complete Enterprise Reports & Export Center Dashboard Page for Phase 19."""
    render_top_header("Enterprise Reports & Export Center")
    render_breadcrumb(["Home", "Governance", "Reports Center"])

    # 1. Sidebar Filters
    filters = render_reports_filter_panel()

    # 2. Reports Search Bar
    st.markdown("### 🔍 Search Enterprise Report Catalog")
    search_query = render_reports_search_input()

    catalog = reports_service.get_report_catalog()

    if search_query.strip():
        q = search_query.strip().lower()
        catalog = [r for r in catalog if q in r["name"].lower() or q in r["category"].lower() or q in " ".join(r["supported_formats"]).lower()]
        st.info(f"Showing {len(catalog)} reports matching search query `{search_query}`.")

    # 3. Active Report Preview Session State Handler
    if "active_preview_id" not in st.session_state:
        st.session_state["active_preview_id"] = None

    # Render Preview Section if active
    if st.session_state["active_preview_id"]:
        _render_report_preview_section(st.session_state["active_preview_id"], filters)
        st.markdown("---")

    # 4. Report Category Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📋 All Reports Catalog",
        "📊 Executive Reports",
        "👥 Customer Reports",
        "🤖 AI Reports",
        "⚙️ Technical Reports",
        "📜 Report History"
    ])

    with t1:
        st.markdown("#### 📋 All Enterprise Reports Catalog")
        _render_report_cards_grid(catalog, filters, prefix="tab_all")

    with t2:
        st.markdown("#### 📊 Executive Reports")
        exec_reps = [r for r in catalog if r["category"] == "Executive"]
        _render_report_cards_grid(exec_reps, filters, prefix="tab_exec")

    with t3:
        st.markdown("#### 👥 Customer Reports")
        cust_reps = [r for r in catalog if r["category"] == "Customer"]
        _render_report_cards_grid(cust_reps, filters, prefix="tab_cust")

    with t4:
        st.markdown("#### 🤖 AI & ML Intelligence Reports")
        ai_reps = [r for r in catalog if r["category"] == "AI"]
        _render_report_cards_grid(ai_reps, filters, prefix="tab_ai")

    with t5:
        st.markdown("#### ⚙️ Technical & MLOps Governance Reports")
        tech_reps = [r for r in catalog if r["category"] == "Technical"]
        _render_report_cards_grid(tech_reps, filters, prefix="tab_tech")

    with t6:
        st.markdown("#### 📜 Generated Report History & Archive")
        _render_report_history_section()

    st.markdown("---")

    # 5. Scheduled Reports Architecture Status
    with st.expander("ℹ️ Scheduled Report Automation Architecture (Disabled)", expanded=False):
        sched_config = reports_service.get_scheduled_report_config()
        st.json(sched_config)


def _render_report_cards_grid(report_list: list, filters: dict, prefix: str = "grid"):
    """Renders a responsive grid of enterprise report cards."""
    if not report_list:
        st.warning("No reports found matching criteria.")
        return

    cols = st.columns(3)
    for i, rep in enumerate(report_list):
        col_idx = i % 3
        with cols[col_idx]:
            icon = "📊" if rep["category"] == "Executive" else ("👥" if rep["category"] == "Customer" else ("🤖" if rep["category"] == "AI" else "⚙️"))
            fmt_badges = " ".join([f"<span class='badge-cyan'>{fmt}</span>" for fmt in rep["supported_formats"]])
            
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 16px; height: 260px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="font-size: 28px;">{icon}</div>
                            <span class="badge-positive">{rep['status']}</span>
                        </div>
                        <div style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-top: 8px;">{rep['name']}</div>
                        <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{rep['description']}</div>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: #64748b; margin-top: 6px;"><b>Period:</b> {rep['data_period']}</div>
                        <div style="margin-top: 6px;">{fmt_badges}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            c_prev, c_gen = st.columns(2)
            with c_prev:
                if st.button("👁️ Preview", key=f"{prefix}_prev_{rep['id']}_{i}"):
                    st.session_state["active_preview_id"] = rep["id"]
                    st.rerun()

            with c_gen:
                if st.button("📥 Export PDF", key=f"{prefix}_pdf_{rep['id']}_{i}"):
                    res = reports_service.generate_and_save_report(rep["id"], export_format="PDF", filter_params=filters)
                    st.success(f"Generated PDF: `{res['filename']}`")
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=res["file_bytes"],
                        file_name=res["filename"],
                        mime="application/pdf",
                        key=f"{prefix}_dl_pdf_{rep['id']}_{i}"
                    )


def _render_report_preview_section(report_id: str, filters: dict):
    """Renders live pre-download report preview panel."""
    st.markdown("### 👁️ Live Report Preview")
    
    col_close, _ = st.columns([1, 5])
    with col_close:
        if st.button("❌ Close Preview"):
            st.session_state["active_preview_id"] = None
            st.rerun()

    preview = reports_service.generate_report_preview(report_id, filters)

    st.markdown(
        f"""
        <div class="glass-card" style="margin-top: 8px; border-left: 4px solid #38bdf8;">
            <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">{preview['report_title']}</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
                <b>Reporting Period:</b> {preview['reporting_period']} | <b>Generated:</b> {preview['generated_date']} | <b>Rows Evaluated:</b> {preview['data_row_count']:,}
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 10px;">
                <b>💡 Executive Summary Note:</b> {preview['executive_summary_text']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Key Metrics Summary Table")
    kpi_rows = []
    for k, v in preview["kpi_summary"].items():
        kpi_rows.append({"KPI Metric": k, "Current Value": v.get("value", "N/A"), "Trend Change": v.get("change_pct", "0.0%")})
    st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True)

    st.markdown("#### Sample Dataset Records (Top 10)")
    sample_df = preview["sample_table"]
    if not sample_df.empty:
        st.dataframe(sample_df.head(10), use_container_width=True)

    st.markdown("#### Instant Export & Generation Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📄 Generate PDF Report"):
            res = reports_service.generate_and_save_report(report_id, export_format="PDF", filter_params=filters)
            st.download_button("⬇️ Download Formatted PDF", data=res["file_bytes"], file_name=res["filename"], mime="application/pdf")
    with c2:
        if st.button("📊 Generate Excel Workbook"):
            res = reports_service.generate_and_save_report(report_id, export_format="EXCEL", filter_params=filters)
            st.download_button("⬇️ Download Excel (.xlsx)", data=res["file_bytes"], file_name=res["filename"], mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        if st.button("📝 Generate CSV Dataset"):
            res = reports_service.generate_and_save_report(report_id, export_format="CSV", filter_params=filters)
            st.download_button("⬇️ Download CSV File", data=res["file_bytes"], file_name=res["filename"], mime="text/csv")


def _render_report_history_section():
    """Renders historical report logging table."""
    history = reports_service.get_report_history()
    if not history:
        st.info("No generated report history logged yet. Generate a report above to populate history.")
        return

    st.markdown(f"Found **{len(history)}** Archived Report Generations")
    st.dataframe(pd.DataFrame(history), use_container_width=True)
