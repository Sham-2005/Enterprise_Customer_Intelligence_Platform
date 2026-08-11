"""
Enterprise MLOps, Model Registry & AI Governance Dashboard Page for ECIP Phase 18.
Wraps render_mlops_layout for backward compatibility.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.mlops_page import render_mlops_layout

def render_mlops_analytics_page(feature_store_df: pd.DataFrame = None):
    """Delegates to Phase 18 MLOps Control Dashboard layout."""
    render_mlops_layout()
