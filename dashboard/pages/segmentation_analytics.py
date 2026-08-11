"""
Customer Segmentation & RFM Intelligence Dashboard Page for ECIP.
Displays 2D/3D PCA cluster scatter plots, business persona cards, algorithm evaluation benchmarks,
and RFM matrix visual components via SegmentationService backend.
"""

import pandas as pd
import streamlit as st
from dashboard.pages.segmentation_page import render_segmentation_layout

def render_segmentation_page(
    segmented_df: pd.DataFrame = None, personas_df: pd.DataFrame = None, benchmark_df: pd.DataFrame = None
):
    """Entry point for rendering Customer Segmentation Page."""
    render_segmentation_layout()
