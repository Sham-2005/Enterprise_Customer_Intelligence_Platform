"""
AI Recommendation Engine & Personalization Dashboard Page for ECIP Phase 16.
Wraps render_recommendation_layout for backward compatibility.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.recommendation_page import render_recommendation_layout

def render_recommendation_analytics_page(
    master_df: pd.DataFrame = None, feature_store_df: pd.DataFrame = None
):
    """Delegates to Phase 16 AI Recommendation Dashboard layout."""
    render_recommendation_layout()
