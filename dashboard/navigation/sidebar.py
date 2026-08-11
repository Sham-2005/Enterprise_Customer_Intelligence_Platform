"""
Collapsible Sidebar Navigation Module for ECIP Dashboard.
Provides icon-rich navigation routing across all executive and AI analytics modules.
"""

import streamlit as st

def render_sidebar_navigation() -> str:
    """Renders collapsible sidebar navigation menu."""
    st.sidebar.markdown(
        """
        <div style="font-size: 20px; font-weight: 900; background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px;">
            ⚡ ECIP PLATFORM
        </div>
        <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;">
            Customer Intelligence Suite
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_route = st.sidebar.radio(
        "Navigation Routes",
        options=[
            "🏠 Executive Dashboard",
            "👥 Customer Analytics",
            "🎯 Customer Segmentation",
            "⚠️ Churn Prediction",
            "💰 Customer Lifetime Value",
            "🤖 Recommendation Engine",
            "🛒 Market Basket Analysis",
            "📈 MLOps Dashboard",
            "📄 Reports"
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("ECIP Platform v1.0.0 | Enterprise Analytics Engine")
    return selected_route
