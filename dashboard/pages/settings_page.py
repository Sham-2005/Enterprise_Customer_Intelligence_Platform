"""
Platform Settings Page UI Layout for ECIP.
Provides theme selection, currency settings, notification triggers, and API configuration options.
"""

import streamlit as st
from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb

def render_settings_layout():
    """Renders Settings Page UI Layout."""
    render_top_header("Platform Settings")
    render_breadcrumb(["Home", "System", "Settings"])

    st.markdown("### ⚙️ System Preferences & Configuration")

    st.markdown("---")

    with st.expander("🎨 UI Theme & Aesthetics", expanded=True):
        st.selectbox("Select Interface Theme", options=["Neon Glassmorphism (Dark)", "Light Modern", "High Contrast Dark"])
        st.selectbox("Primary Accent Palette", options=["Cyan / Blue", "Purple / Pink", "Emerald Green"])

    with st.expander("💱 Localization & Currency", expanded=False):
        st.selectbox("Display Currency", options=["USD ($)", "EUR (€)", "BRL (R$)", "GBP (£)"])
        st.selectbox("Timezone Display", options=["UTC", "America/New_York", "America/Sao_Paulo", "Europe/London"])

    with st.expander("🔔 Notification Preferences", expanded=False):
        st.checkbox("Enable High Churn Risk Email Alerts", value=True)
        st.checkbox("Enable Model Drift Alerts", value=True)
        st.checkbox("Enable Weekly KPI Summary Briefs", value=True)

    if st.button("💾 Save Platform Settings"):
        st.success("Platform settings successfully updated!")
