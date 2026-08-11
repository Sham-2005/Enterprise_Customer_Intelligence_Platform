"""
Customer Analytics Dashboard Page for ECIP.
Displays Customer growth, active vs inactive ratios, RFM segments, CLV distributions,
state maps, and reports exports via CustomerAnalyticsService.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.customer_analytics_page import render_customer_analytics_layout

def render_customer_analytics_page(feature_store_df: pd.DataFrame = None):
    """Entry point for rendering Customer Analytics Page."""
    render_customer_analytics_layout()
