"""
Export Utility Module for ECIP Dashboard.
Provides triggers to export filtered DataFrames into CSV and Excel formats.
"""

import io
import pandas as pd
import streamlit as st

def render_export_buttons(df: pd.DataFrame, filename_prefix: str = "ecip_export"):
    """Renders download buttons for CSV and Excel formats in Streamlit UI."""
    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv"
        )

    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        buffer.seek(0)
        st.download_button(
            label="📊 Export Excel",
            data=buffer,
            file_name=f"{filename_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
