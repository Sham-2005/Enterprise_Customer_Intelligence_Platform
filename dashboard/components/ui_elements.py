"""
Reusable UI Elements & State Feedback Components for ECIP.
Provides Breadcrumbs, Status Badges, Empty State Placeholders, Loading Spinners, and Error Cards.
"""

import streamlit as st

def render_breadcrumb(path_items: list[str]):
    """Renders styled breadcrumbs."""
    items_html = " &nbsp;&nbsp;/&nbsp;&nbsp; ".join(
        [f"<span style='color: #94a3b8;'>{item}</span>" for item in path_items[:-1]] +
        [f"<span style='color: #38bdf8; font-weight: 700;'>{path_items[-1]}</span>"]
    )
    st.markdown(f"<div style='font-size: 13px; margin-bottom: 12px;'>🏠 {items_html}</div>", unsafe_allow_html=True)

def render_status_badge(text: str, status_type: str = "success"):
    """Renders status badge pills."""
    color_map = {
        "success": ("rgba(52, 211, 153, 0.15)", "#34d399", "rgba(52, 211, 153, 0.3)"),
        "danger": ("rgba(248, 113, 113, 0.15)", "#f87171", "rgba(248, 113, 113, 0.3)"),
        "warning": ("rgba(251, 191, 36, 0.15)", "#fbbf24", "rgba(251, 191, 36, 0.3)"),
        "info": ("rgba(56, 189, 248, 0.15)", "#38bdf8", "rgba(56, 189, 248, 0.3)"),
        "purple": ("rgba(168, 85, 247, 0.15)", "#c084fc", "rgba(168, 85, 247, 0.3)")
    }
    bg, fg, border = color_map.get(status_type, color_map["info"])
    badge_html = f"<span style='background: {bg}; color: {fg}; border: 1px solid {border}; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700;'>{text}</span>"
    return badge_html

def render_empty_state(title: str = "No Data Available", message: str = "Please adjust filter parameters or load processed dataset artifacts."):
    """Renders glassmorphic empty state placeholder."""
    html = f"""
    <div class="glass-card" style="text-align: center; padding: 40px !important;">
        <div style="font-size: 40px; margin-bottom: 10px;">📦</div>
        <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">{title}</div>
        <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
