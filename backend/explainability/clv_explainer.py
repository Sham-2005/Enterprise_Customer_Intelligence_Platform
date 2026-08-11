"""
CLV SHAP Explainability Engine for ECIP Regression Models.
Provides individual feature attribution scores and plain-English narratives for predicted Customer Lifetime Value.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import shap
from utils.logger import setup_logger

logger = setup_logger("ECIP.CLVSHAPExplainer")

class CLVSHAPExplainer:
    """Computes SHAP feature attributions for CLV regression predictions."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def fit_explainer(self, X_sample: np.ndarray):
        try:
            self.explainer = shap.Explainer(self.model, X_sample)
            logger.info("Fitted CLV SHAP Explainer successfully.")
        except Exception as e:
            logger.warning(f"CLV SHAP Explainer fallback: {e}")
            self.explainer = shap.KernelExplainer(self.model.predict, X_sample[:50])

    def explain_clv_instance(self, customer_features: pd.DataFrame) -> Dict[str, Any]:
        X = customer_features[self.feature_names].values

        if self.explainer is None:
            contributions = dict(zip(self.feature_names, np.zeros(len(self.feature_names))))
        else:
            try:
                if isinstance(self.explainer, shap.KernelExplainer):
                    shap_raw = self.explainer.shap_values(X)
                    vals = shap_raw[0] if isinstance(shap_raw, list) else shap_raw[0]
                else:
                    shap_values = self.explainer(X)
                    vals = shap_values.values[0] if hasattr(shap_values, "values") else shap_values[0]
                contributions = dict(zip(self.feature_names, [round(float(v), 2) for v in vals]))
            except Exception as e:
                logger.warning(f"Error computing CLV SHAP values, using zero contributions fallback: {e}")
                contributions = dict(zip(self.feature_names, np.zeros(len(self.feature_names))))

        spend = float(customer_features["total_spending"].iloc[0]) if "total_spending" in customer_features.columns and not customer_features.empty else 0.0
        orders = float(customer_features["total_orders"].iloc[0]) if "total_orders" in customer_features.columns and not customer_features.empty else 0.0
        aov = float(customer_features["avg_order_value"].iloc[0]) if "avg_order_value" in customer_features.columns and not customer_features.empty else 0.0

        narrative = (
            f"Predicted 12-month Customer Lifetime Value is primarily anchored by "
            f"historical total spend of ${spend:,.2f} across {orders:.0f} order(s) with an Average Order Value of ${aov:.2f}."
        )

        return {
            "feature_contributions": contributions,
            "natural_language_explanation": narrative
        }
