"""
Business Intelligence (Power BI) Integration Module for ECIP Dashboard.
Renders responsive Power BI Service embedded reports or enterprise setup blueprints.
"""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb

# Optional PyYAML import with fallback
try:
    import yaml
    HAS_YAML = True
except (ImportError, ModuleNotFoundError):
    HAS_YAML = False

DEFAULT_POWERBI_CONFIG = {
    "reports": {
        "executive": {
            "title": "Executive Summary Dashboard",
            "embed_url": "",
            "web_url": "https://app.powerbi.com/groups/me/reports/sample-executive",
            "description": "C-suite high-level revenue trajectory, gross margin, and executive KPI summary."
        },
        "customer": {
            "title": "Customer Analytics & Demographics",
            "embed_url": "",
            "web_url": "https://app.powerbi.com/groups/me/reports/sample-customer",
            "description": "Geographic buyer concentration, repeat buyer rates, and customer acquisition cohorts."
        },
        "sales": {
            "title": "Sales & Revenue Performance",
            "embed_url": "",
            "web_url": "https://app.powerbi.com/groups/me/reports/sample-sales",
            "description": "Revenue breakdown by payment type, installment plan, and seasonal order volumes."
        },
        "product": {
            "title": "Product Catalog & Merchandising",
            "embed_url": "",
            "web_url": "https://app.powerbi.com/groups/me/reports/sample-product",
            "description": "Pareto 80/20 category revenue analysis, freight cost impact, and SKU velocity."
        },
        "ai_analytics": {
            "title": "AI Risk & Predictive Analytics",
            "embed_url": "",
            "web_url": "https://app.powerbi.com/groups/me/reports/sample-ai",
            "description": "Churn risk distributions, 12M CLV value tiering, and market basket association rules."
        }
    }
}

def load_powerbi_config() -> dict:
    """Loads Power BI YAML configuration file with safe fallback."""
    if HAS_YAML:
        config_path = Path(__file__).resolve().parent.parent.parent / "powerbi" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if loaded and isinstance(loaded, dict):
                        return loaded
            except Exception:
                pass
    return DEFAULT_POWERBI_CONFIG

def render_powerbi_report_card(report_key: str, report_data: dict):
    """Renders Power BI embedded iframe or glassmorphic fallback setup blueprint."""
    title = report_data.get("title", report_key.capitalize())
    description = report_data.get("description", "Enterprise Power BI analytical report.")
    embed_url = report_data.get("embed_url", "").strip() if report_data.get("embed_url") else ""
    web_url = report_data.get("web_url", "https://app.powerbi.com")

    st.markdown(f"### 📊 {title}")
    st.caption(f"💡 {description}")
    st.markdown("---")

    if embed_url:
        # Full-width responsive Power BI iframe embed
        components.iframe(embed_url, height=750, scrolling=True)
    else:
        # Enterprise Glassmorphic Setup Blueprint Card
        html_blueprint = f"""
        <div class="glass-card" style="padding: 30px !important;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="font-size: 20px; font-weight: 800; color: #f8fafc;">⚡ Power BI Report Integration Mode</div>
                    <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">{title} is ready for Power BI Service live embedding.</div>
                </div>
                <span class="badge-purple">STATUS: CONFIGURATION READY</span>
            </div>
            
            <hr style="border-color: rgba(255,255,255,0.08); margin: 18px 0;">

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin-bottom: 8px;">🛠️ Integration Instructions</div>
                    <ol style="color: #cbd5e1; font-size: 12px; line-height: 1.8; margin-left: 16px; padding-left: 0;">
                        <li>Open Power BI Desktop and load <code>output/master_dataset.csv</code>.</li>
                        <li>Build executive report visuals or import <code>powerbi/config.yaml</code> templates.</li>
                        <li>Publish report to your <b>Power BI Service Workspace</b>.</li>
                        <li>Copy Embed URL into <code>powerbi/config.yaml</code> under <code>embed_url</code>.</li>
                    </ol>
                </div>

                <div style="background: rgba(15, 23, 42, 0.6); padding: 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 14px; font-weight: 700; color: #c084fc; margin-bottom: 8px;">📑 Snapshot Preview Capabilities</div>
                    <p style="color: #cbd5e1; font-size: 12px; line-height: 1.6;">
                        This module supports direct Web Service linking, exported PDF snapshots, and DAX analytical measure definitions.
                    </p>
                    <div style="margin-top: 14px;">
                        <span class="badge-cyan">DAX Engine</span>
                        <span class="badge-positive" style="margin-left: 6px;">DirectQuery</span>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(html_blueprint, unsafe_allow_html=True)

        # Action Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.link_button("🚀 Launch Power BI Service Web Link", web_url, use_container_width=True)
        with col2:
            st.button("📥 View Setup Guide (PowerBI_Setup_Guide.md)", key=f"guide_{report_key}", use_container_width=True)
        with col3:
            st.button("📄 Export Report Snapshot (PDF)", key=f"snapshot_{report_key}", use_container_width=True)

def render_powerbi_layout():
    """Renders main Power BI page layout with tab selector."""
    render_top_header("Business Intelligence (Power BI)")
    render_breadcrumb(["Home", "Executive Reporting", "Business Intelligence (Power BI)"])

    cfg = load_powerbi_config().get("reports", {})

    # Tab Selector for Multiple Reports
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Executive Summary",
        "👥 Customer Analytics",
        "💳 Sales Analytics",
        "🛍️ Product Analytics",
        "🤖 AI Analytics"
    ])

    with t1:
        render_powerbi_report_card("executive", cfg.get("executive", {}))
    with t2:
        render_powerbi_report_card("customer", cfg.get("customer", {}))
    with t3:
        render_powerbi_report_card("sales", cfg.get("sales", {}))
    with t4:
        render_powerbi_report_card("product", cfg.get("product", {}))
    with t5:
        render_powerbi_report_card("ai_analytics", cfg.get("ai_analytics", {}))
