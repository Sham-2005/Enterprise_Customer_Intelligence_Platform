"""
Platform Overview Component Library for ECIP Dashboard.
Renders enterprise KPI cards for Platform Overview in a responsive grid layout.
"""

from typing import Dict, Any, List
import streamlit as st

def render_platform_overview_card(
    title: str,
    value: str,
    trend: str = "↑ +12.4%",
    is_positive: bool = True,
    description: str = "Primary metric indicator",
    icon: str = "📊",
    accent_color: str = "cyan"
):
    """Renders a single modern glassmorphic enterprise KPI card with hover animation."""
    badge_class = "badge-positive" if is_positive else "badge-negative"
    if accent_color == "cyan":
        badge_class = "badge-cyan"
    elif accent_color == "purple":
        badge_class = "badge-purple"

    card_html = f"""
    <div class="glass-card" style="margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div class="kpi-title">{title}</div>
            <div style="font-size: 22px; opacity: 0.85;">{icon}</div>
        </div>
        <div class="kpi-value" style="font-size: 26px;">{value}</div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 12px; gap: 8px;">
            <span class="{badge_class}">{trend}</span>
            <span style="font-size: 11px; color: #94a3b8; text-align: right; line-height: 1.2;">{description}</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_platform_overview_section():
    """Renders the 8-card Platform Overview grid section."""
    st.markdown("### 🌐 Platform Overview & Enterprise Intelligence")
    st.markdown("Real-time executive performance summary across operational, predictive ML, and CSAT dimensions.")
    st.markdown("---")

    overview_metrics = [
        {"title": "Total Customers", "value": "96,096", "trend": "↑ +11.5%", "is_positive": True, "description": "Active registered buyers", "icon": "👥", "accent_color": "cyan"},
        {"title": "Total Orders", "value": "99,441", "trend": "↑ +8.7%", "is_positive": True, "description": "Fulfilled e-commerce orders", "icon": "📦", "accent_color": "green"},
        {"title": "Total Revenue", "value": "$16,008,872.12", "trend": "↑ +14.2%", "is_positive": True, "description": "Gross sales volume", "icon": "💰", "accent_color": "purple"},
        {"title": "Total Products", "value": "32,951", "trend": "↑ +4.8%", "is_positive": True, "description": "Active catalog SKUs", "icon": "🛍️", "accent_color": "cyan"},
        {"title": "Active AI Models", "value": "5 / 5 Active", "trend": "↑ 100%", "is_positive": True, "description": "MLOps registered models", "icon": "🤖", "accent_color": "purple"},
        {"title": "Total Predictions", "value": "1,248,900", "trend": "↑ +22.4%", "is_positive": True, "description": "Real-time ML inferences", "icon": "📊", "accent_color": "green"},
        {"title": "Average Customer Rating", "value": "4.15 / 5.0", "trend": "↑ +0.12", "is_positive": True, "icon": "⭐", "description": "CSAT rating score", "accent_color": "cyan"},
        {"title": "Business Health Score", "value": "94.8 / 100", "trend": "↑ +2.5%", "is_positive": True, "description": "Composite health index", "icon": "📈", "accent_color": "green"}
    ]

    # Row 1 (4 Columns)
    r1_cols = st.columns(4)
    for idx in range(4):
        m = overview_metrics[idx]
        with r1_cols[idx]:
            render_platform_overview_card(
                title=m["title"],
                value=m["value"],
                trend=m["trend"],
                is_positive=m["is_positive"],
                description=m["description"],
                icon=m["icon"],
                accent_color=m["accent_color"]
            )

    # Row 2 (4 Columns)
    r2_cols = st.columns(4)
    for idx in range(4, 8):
        m = overview_metrics[idx]
        with r2_cols[idx - 4]:
            render_platform_overview_card(
                title=m["title"],
                value=m["value"],
                trend=m["trend"],
                is_positive=m["is_positive"],
                description=m["description"],
                icon=m["icon"],
                accent_color=m["accent_color"]
            )
