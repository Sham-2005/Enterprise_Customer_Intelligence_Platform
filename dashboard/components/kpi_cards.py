"""
Glassmorphic KPI Cards & Metric Component Library for ECIP Dashboard.
Renders executive metric containers with percentage changes, glowing borders,
trend arrows, previous period baselines, and timestamps.
"""

from typing import Dict, Any, List
import streamlit as st

def render_glass_kpi_card(
    title: str,
    value: str,
    change_pct: str = "↑ +12.4%",
    is_positive: bool = True,
    subtext: str = "vs previous period",
    icon: str = "📊",
    badge_type: str = "green",
    last_updated: str = ""
):
    """Renders a single glassmorphic KPI card with period metrics and timestamp."""
    badge_class = "badge-positive" if is_positive else "badge-negative"
    if badge_type == "cyan":
        badge_class = "badge-cyan"
    elif badge_type == "purple":
        badge_class = "badge-purple"

    ts_html = f'<div style="font-size: 10px; color: #64748b; margin-top: 6px;">🕒 {last_updated}</div>' if last_updated else ''

    card_html = f"""
    <div class="glass-card" style="margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div class="kpi-title">{title}</div>
            <div style="font-size: 20px; opacity: 0.85;">{icon}</div>
        </div>
        <div class="kpi-value" style="font-size: 24px; font-weight: 700; margin: 8px 0;">{value}</div>
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap;">
            <span class="{badge_class}">{change_pct}</span>
            <span style="font-size: 11px; color: #94a3b8; text-align: right;">{subtext}</span>
        </div>
        {ts_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_kpi_card_from_dict(metric: Dict[str, Any]):
    """Helper to render a KPI card directly from KPIService metric dictionary."""
    render_glass_kpi_card(
        title=metric.get("title", "Metric"),
        value=metric.get("value", "0"),
        change_pct=metric.get("change_pct", "↑ 0.0%"),
        is_positive=metric.get("is_positive", True),
        subtext=metric.get("subtext", "vs prev period"),
        icon=metric.get("icon", "📊"),
        badge_type="green" if metric.get("is_positive", True) else "red",
        last_updated=metric.get("last_updated", "")
    )

def render_kpi_grid_row(kpis: List[Dict[str, Any]]):
    """Renders a row of KPI cards dynamically based on column count."""
    cols = st.columns(len(kpis))
    for idx, kpi in enumerate(kpis):
        with cols[idx]:
            render_kpi_card_from_dict(kpi)
