"""
Customer Lifetime Value (CLV) & Revenue Intelligence Dashboard Page for ECIP.
Displays projected system CLV metrics, high-value customer leaderboards, 12-month revenue forecasts,
and interactive real-time prediction sandboxes via CLVService backend.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.clv_page import render_clv_layout

def render_clv_analytics_page(
    clv_predictions_df: pd.DataFrame = None, high_value_df: pd.DataFrame = None, forecast_df: pd.DataFrame = None
):
    """Entry point for rendering Customer Lifetime Value (CLV) Page."""
    render_clv_layout()
