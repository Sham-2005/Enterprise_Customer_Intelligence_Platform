"""
Data Quality Report Generator for ECIP.
Compiles validation metrics, feature summaries, duplicate stats, missing value distributions,
and Data Quality Score into HTML and CSV report exports.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.QualityReporter")

class DataQualityReporter:
    """Generates standalone HTML and CSV Data Quality & Data Hygiene reports."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.reports_dir = self.settings.get_path("paths.reports_dir")

    def generate_report(
        self, validation_results: Dict[str, Any], feature_store_df: pd.DataFrame
    ) -> Tuple[Path, Path]:
        """
        Generates and saves HTML and CSV Data Quality Reports.

        Returns:
            Tuple[html_report_path, csv_report_path]
        """
        logger.info("Generating Data Quality Report...")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        quality_score, summary_df = self._build_metrics_summary(validation_results, feature_store_df)

        html_path = self.reports_dir / "data_quality_report.html"
        csv_path = self.reports_dir / "data_quality_report.csv"

        # Save CSV report
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV Data Quality Report to {csv_path}")

        # Save HTML report
        html_content = self._build_html_markup(quality_score, summary_df, feature_store_df)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved HTML Data Quality Report to {html_path}")

        return html_path, csv_path

    def _build_metrics_summary(
        self, validation_results: Dict[str, Any], feature_store_df: pd.DataFrame
    ) -> Tuple[float, pd.DataFrame]:
        rows_list = []
        total_cells = 0
        total_missing = 0
        total_duplicates = 0

        for dataset_name, stats in validation_results.items():
            r = stats["total_rows"]
            c = stats["total_columns"]
            dups = stats["duplicate_rows"]
            nulls = stats["missing_cells"]

            total_cells += (r * c)
            total_missing += nulls
            total_duplicates += dups

            rows_list.append({
                "Dataset": dataset_name,
                "Rows": r,
                "Columns": c,
                "Duplicate Rows": dups,
                "Missing Cells": nulls,
                "Duplicate Rate (%)": round((dups / max(r, 1)) * 100, 2),
                "Missing Cell Rate (%)": round((nulls / max(r * c, 1)) * 100, 2)
            })

        summary_df = pd.DataFrame(rows_list)

        # Compute Overall Data Quality Score (100 minus weighted penalties for missing/duplicates)
        missing_rate = (total_missing / max(total_cells, 1)) * 100
        dup_rate = (total_duplicates / max(sum(s["total_rows"] for s in validation_results.values()), 1)) * 100
        quality_score = max(round(100.0 - (missing_rate * 2.0 + dup_rate * 3.0), 2), 0.0)

        return quality_score, summary_df

    def _build_html_markup(
        self, quality_score: float, summary_df: pd.DataFrame, feature_store_df: pd.DataFrame
    ) -> str:
        table_html = summary_df.to_html(classes="styled-table", index=False)
        feature_count = len(feature_store_df.columns)
        customer_count = len(feature_store_df)

        feature_summary_sample = pd.DataFrame({
            "Feature Name": feature_store_df.columns[:15],
            "Data Type": [str(feature_store_df[col].dtype) for col in feature_store_df.columns[:15]],
            "Null Count": [int(feature_store_df[col].isnull().sum()) for col in feature_store_df.columns[:15]],
            "Unique Values": [int(feature_store_df[col].nunique()) for col in feature_store_df.columns[:15]]
        }).to_html(classes="styled-table", index=False)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ECIP - Data Quality & Feature Store Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }}
        h1, h2 {{ color: #38bdf8; font-weight: 600; }}
        .metric-card {{ background: #1e293b; padding: 20px; border-radius: 12px; display: inline-block; margin-right: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #4ade80; }}
        .metric-label {{ color: #94a3b8; font-size: 14px; text-transform: uppercase; }}
        .styled-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 14px; text-align: left; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        .styled-table th {{ background-color: #0284c7; color: #ffffff; padding: 12px 15px; }}
        .styled-table td {{ padding: 12px 15px; border-bottom: 1px solid #334155; color: #e2e8f0; }}
        .styled-table tr:nth-of-type(even) {{ background-color: #1e293b; }}
        .styled-table tr:nth-of-type(odd) {{ background-color: #0f172a; }}
    </style>
</head>
<body>
    <h1>Enterprise Customer Intelligence Platform (ECIP)</h1>
    <h2>Automated Data Quality & Feature Store Audit Report</h2>
    <hr style="border-color: #334155; margin-bottom: 30px;">

    <div>
        <div class="metric-card">
            <div class="metric-label">Data Quality Score</div>
            <div class="metric-value">{quality_score}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Processed Customers</div>
            <div class="metric-value">{customer_count:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Engineered Features</div>
            <div class="metric-value">{feature_count}</div>
        </div>
    </div>

    <h2>Raw Dataset Ingestion Audit</h2>
    {table_html}

    <h2>Feature Store Sample Metrics (First 15 Features)</h2>
    {feature_summary_sample}

    <p style="color: #64748b; margin-top: 40px;">Report auto-generated by ECIP QualityReporter Engine.</p>
</body>
</html>
"""
        return html
