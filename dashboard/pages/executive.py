"""
Executive Overview Dashboard Page for ECIP.
Displays C-suite KPI metrics, revenue performance, monthly growth, sales trends,
and reports exports via ExecutiveDashboardBackend services.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.executive_page import render_executive_dashboard_layout

def render_executive_page(master_df: pd.DataFrame = None, feature_store_df: pd.DataFrame = None):
    """Entry point for rendering Executive Dashboard."""
    render_executive_dashboard_layout()
