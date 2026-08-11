"""
Enterprise Customer Intelligence Platform (ECIP) - BI Dashboard Entrypoint.
Glassmorphism Dark Theme Interface with Modular Navigation Router.
"""

import sys
from pathlib import Path

# Add project root to sys.path at index 0 before other imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from dashboard.styles.glassmorphism import apply_glassmorphism_theme
from dashboard.navigation.sidebar import render_sidebar_navigation
from dashboard.pages.executive_page import render_executive_dashboard_layout
from dashboard.pages.customer_analytics_page import render_customer_analytics_layout
from dashboard.pages.segmentation_page import render_segmentation_layout
from dashboard.pages.churn_page import render_churn_layout
from dashboard.pages.clv_page import render_clv_layout
from dashboard.pages.recommendation_page import render_recommendation_layout
from dashboard.pages.mba_page import render_mba_layout
from dashboard.pages.mlops_page import render_mlops_layout
from dashboard.pages.reports_page import render_reports_layout

def main():
    st.set_page_config(
        page_title="ECIP - Enterprise Intelligence Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Step 1: Inject Glassmorphism CSS Design System
    apply_glassmorphism_theme()

    # Step 2: Render Collapsible Sidebar Navigation Menu
    selected_route = render_sidebar_navigation()

    # Step 3: Route to Selected UI Page Layout
    if selected_route == "🏠 Executive Dashboard":
        render_executive_dashboard_layout()

    elif selected_route == "👥 Customer Analytics":
        render_customer_analytics_layout()

    elif selected_route == "🎯 Customer Segmentation":
        render_segmentation_layout()

    elif selected_route == "⚠️ Churn Prediction":
        render_churn_layout()

    elif selected_route == "💰 Customer Lifetime Value":
        render_clv_layout()

    elif selected_route == "🤖 Recommendation Engine":
        render_recommendation_layout()

    elif selected_route == "🛒 Market Basket Analysis":
        render_mba_layout()

    elif selected_route == "📈 MLOps Dashboard":
        render_mlops_layout()

    elif selected_route == "📄 Reports":
        render_reports_layout()

if __name__ == "__main__":
    main()
