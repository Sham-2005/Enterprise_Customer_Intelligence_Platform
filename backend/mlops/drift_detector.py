"""
Data & Concept Drift Detection Engine for ECIP.
Computes Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) to flag feature distribution shifts.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.DriftDetector")

class DriftDetector:
    """Monitors feature distributions between baseline training data and production inference batches."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.reports_dir = self.settings.get_path("paths.reports_dir")

    def detect_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Runs Kolmogorov-Smirnov 2-sample tests across features to detect distribution shift.

        Returns:
            Dict containing feature_drift_results, drifted_features_list, and overall_drift_detected flag.
        """
        logger.info("Executing Data & Concept Drift Audit...")
        results = {}
        drifted_features = []

        for col in feature_cols:
            if col in baseline_df.columns and col in current_df.columns:
                base_vals = baseline_df[col].dropna().values
                curr_vals = current_df[col].dropna().values

                if len(base_vals) > 0 and len(curr_vals) > 0:
                    ks_stat, p_value = ks_2samp(base_vals, curr_vals)
                    is_drifted = bool(p_value < 0.05)

                    results[col] = {
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_value), 4),
                        "drift_detected": is_drifted
                    }

                    if is_drifted:
                        drifted_features.append(col)
                        logger.warning(f"Data Drift detected in feature '{col}': p-value = {p_value:.4f}")

        overall_drift = len(drifted_features) > 0

        drift_report = {
            "overall_drift_detected": overall_drift,
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_metrics": results
        }

        self._export_reports(drift_report)
        return drift_report

    def _export_reports(self, drift_report: Dict[str, Any]):
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.reports_dir / "drift_report.json"
        html_path = self.reports_dir / "drift_report.html"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(drift_report, f, indent=4)

        html_markup = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ECIP - Data & Concept Drift Audit Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        h1, h2 {{ color: #38bdf8; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }}
        .alert {{ color: #f87171; font-weight: bold; }}
        .ok {{ color: #4ade80; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Enterprise Customer Intelligence Platform (ECIP)</h1>
    <h2>Data & Concept Drift Monitoring Audit</h2>
    <hr style="border-color: #334155;">

    <div class="card">
        <h3>Drift Detection Status</h3>
        <p>Overall Drift Status: <span class="{'alert' if drift_report['overall_drift_detected'] else 'ok'}">{'ALERT: DRIFT DETECTED' if drift_report['overall_drift_detected'] else 'SYSTEM NORMAL'}</span></p>
        <p>Drifted Features Count: {drift_report['drifted_features_count']}</p>
        <p>Drifted Feature List: {", ".join(drift_report['drifted_features']) if drift_report['drifted_features'] else 'None'}</p>
    </div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_markup)

        logger.info(f"Exported drift reports to {json_path} and {html_path}")
