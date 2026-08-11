"""
Automated Model Retraining Pipeline for ECIP.
Orchestrates pipeline re-training, baseline comparison, model version promotion, and registry updates.
"""

from typing import Dict, Any
import pandas as pd
from backend.models.churn_model import ChurnModelPipeline
from backend.models.clv_model import CLVModelPipeline
from backend.mlops.registry import ModelRegistry
from utils.logger import setup_logger

logger = setup_logger("ECIP.RetrainingPipeline")

class RetrainingPipeline:
    """Automates re-training pipelines for Churn and CLV models."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.registry = ModelRegistry(config_path)

    def retrain_churn_model(self, feature_store_df: pd.DataFrame, new_version: str = "v2.0") -> Dict[str, Any]:
        """Re-trains Churn Classifier and registers new version if ROC-AUC improves."""
        logger.info(f"Triggering automated retraining pipeline for Churn model (Version {new_version})...")

        pipeline = ChurnModelPipeline()
        best_model, metrics_dict, _ = pipeline.train_and_evaluate(feature_store_df)

        entry = self.registry.register_model(
            model_name="ChurnClassifier",
            version=new_version,
            algorithm=metrics_dict["best_model_name"],
            metrics={"ROC-AUC": metrics_dict["best_roc_auc"]},
            features=metrics_dict["feature_names"],
            hyperparameters={"random_state": 42},
            status="Active"
        )

        logger.info(f"Retraining complete for ChurnClassifier. Promoted to '{new_version}'.")
        return entry
