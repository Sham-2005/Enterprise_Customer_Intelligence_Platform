"""
Customer Churn Prediction & Risk Intelligence Dashboard Page for ECIP.
Displays risk metrics, interactive real-time prediction sandbox with SHAP explanations,
high-risk customer drilldowns, and batch CSV predictions via ChurnService backend.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.churn_page import render_churn_layout

def render_churn_analytics_page(
    predictions_df: pd.DataFrame = None, high_risk_df: pd.DataFrame = None
):
    """Entry point for rendering Customer Churn Prediction Page."""
    render_churn_layout()
