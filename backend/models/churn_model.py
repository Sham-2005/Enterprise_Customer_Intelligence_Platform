"""
AI Customer Churn Prediction Machine Learning Model Pipeline for ECIP.
Implements feature preprocessing, SMOTE class imbalance handling, multi-model evaluation
(Logistic Regression, Random Forest, XGBoost, LightGBM), hyperparameter optimization, and model serialization.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config.settings import Settings
from utils.logger import setup_logger
from utils.exceptions import ModelTrainingError

logger = setup_logger("ECIP.ChurnModelPipeline")

class ChurnModelPipeline:
    """Enterprise Machine Learning Pipeline for Customer Churn Classification."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.models_dir = self.settings.get_path("paths.models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = ""
        self.feature_names = []

    def train_and_evaluate(
        self, feature_store_df: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
        """
        Executes data splitting, SMOTE balancing, model benchmarking, hyperparameter tuning, and serialization.

        Returns:
            Tuple[best_model_instance, evaluation_metrics_dict, comparison_table_df]
        """
        logger.info("Starting Churn Prediction Model Training & Benchmark Pipeline...")

        if "churn_label" not in feature_store_df.columns:
            raise ModelTrainingError("Target column 'churn_label' missing from Feature Store DataFrame.")

        # 1. Feature Selection
        feature_cols = [
            "total_spending", "total_orders", "avg_order_value",
            "recency_days", "historical_clv", "avg_review_score_given",
            "distinct_categories_count", "loyalty_score", "avg_freight_cost",
            "is_repeat_customer", "purchase_frequency_monthly"
        ]
        self.feature_names = [c for c in feature_cols if c in feature_store_df.columns]
        logger.info(f"Training on {len(self.feature_names)} features: {self.feature_names}")

        X = feature_store_df[self.feature_names].fillna(feature_store_df[self.feature_names].median())
        y = feature_store_df["churn_label"].astype(int)

        # 2. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 4. Handle Class Imbalance via SMOTE or Class Weighting
        if HAS_SMOTE and y_train.nunique() > 1 and y_train.value_counts().min() > 5:
            logger.info("Applying SMOTE oversampling to balance churn classes...")
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        else:
            logger.info("SMOTE unavailable or insufficient class samples; using original stratified class distribution...")
            X_train_res, y_train_res = X_train_scaled, y_train

        # 5. Define Models Pool
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        }

        if HAS_XGB:
            scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=100, random_state=42, eval_metric="logloss",
                scale_pos_weight=scale_pos_weight
            )

        if HAS_LGB:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=100, random_state=42, verbose=-1, class_weight="balanced"
            )

        # 6. Model Evaluation Benchmark
        results = []
        best_auc = -1.0

        for name, model in models.items():
            logger.info(f"Training and evaluating {name}...")
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)

            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall_curve, precision_curve)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4)
            })

            if roc_auc > best_auc:
                best_auc = roc_auc
                self.best_model = model
                self.best_model_name = name

        comparison_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
        logger.info(f"Top performing churn model: {self.best_model_name} (ROC-AUC: {best_auc:.4f})")

        # 7. Serialize Trained Model & Metrics
        metrics_dict = {
            "best_model_name": self.best_model_name,
            "best_roc_auc": round(best_auc, 4),
            "feature_names": self.feature_names,
            "benchmark_results": results
        }
        
        self.save_model(metrics_dict)

        return self.best_model, metrics_dict, comparison_df

    def predict_churn_probability(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Generates churn probabilities for a given feature matrix."""
        if self.best_model is None:
            model_path = self.models_dir / "churn_model.pkl"
            if model_path.exists():
                self.best_model = joblib.load(model_path)
            else:
                raise ModelTrainingError("No trained churn model available.")

        X = feature_df[self.feature_names].fillna(feature_df[self.feature_names].median())
        X_scaled = self.scaler.transform(X)
        probs = self.best_model.predict_proba(X_scaled)[:, 1]
        return probs

    def save_model(self, metrics_dict: Dict[str, Any]):
        model_path = self.models_dir / "churn_model.pkl"
        scaler_path = self.models_dir / "scaler.pkl"
        metrics_path = self.models_dir / "model_metrics.json"

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)

        logger.info(f"Saved trained model to {model_path} and metrics to {metrics_path}")
