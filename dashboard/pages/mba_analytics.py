"""
Market Basket Analysis & Association Rule Mining Dashboard Page for ECIP Phase 17.
Wraps render_mba_layout for backward compatibility.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.mba_page import render_mba_layout

def render_mba_analytics_page(master_df: pd.DataFrame = None):
    """Delegates to Phase 17 Market Basket Analysis Dashboard layout."""
    render_mba_layout()
