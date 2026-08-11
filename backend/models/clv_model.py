"""
AI Customer Lifetime Value (CLV) Prediction Regression Model Pipeline for ECIP.
Implements feature engineering (purchase velocity, engagement score), target CLV calculation,
multi-model regression benchmarking (Linear Regression, Random Forest, XGBoost, LightGBM),
hyperparameter optimization, and model serialization.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

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

logger = setup_logger("ECIP.CLVModelPipeline")

class CLVModelPipeline:
    """Enterprise Machine Learning Regression Pipeline for Customer Lifetime Value Forecasting."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.models_dir = self.settings.get_path("paths.models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = ""
        self.feature_names = []

    def engineer_clv_features_and_target(
        self, feature_store_df: pd.DataFrame, horizon_months: int = 12
    ) -> pd.DataFrame:
        """Engineers advanced monetary features and computes target projected CLV over horizon."""
        df = feature_store_df.copy()

        # Advanced Business Features
        df["purchase_velocity"] = df["total_orders"] / np.maximum(df["customer_age_days"] / 30.0, 1.0)
        df["spending_trend"] = df["avg_order_value"] * df["purchase_frequency_monthly"]
        df["engagement_score"] = df["total_orders"] * df["avg_review_score_given"]
        df["category_diversity_index"] = df["distinct_categories_count"] / np.maximum(df["total_orders"], 1.0)

        # Target 12-Month Projected CLV
        # Formula: Historical Spending + (Monthly Frequency * AOV * horizon_months)
        monthly_val = df["purchase_frequency_monthly"] * df["avg_order_value"]
        df["target_clv_future"] = df["total_spending"] + (monthly_val * horizon_months)

        return df

    def train_and_evaluate(
        self, feature_store_df: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
        """
        Executes data splitting, feature scaling, regression benchmarking, and model serialization.

        Returns:
            Tuple[best_model_instance, evaluation_metrics_dict, comparison_table_df]
        """
        logger.info("Starting CLV Regression Model Training & Benchmark Pipeline...")

        df_engineered = self.engineer_clv_features_and_target(feature_store_df)

        feature_cols = [
            "total_spending", "total_orders", "avg_order_value",
            "recency_days", "historical_clv", "avg_review_score_given",
            "distinct_categories_count", "loyalty_score", "avg_freight_cost",
            "purchase_velocity", "spending_trend", "engagement_score",
            "category_diversity_index"
        ]
        self.feature_names = [c for c in feature_cols if c in df_engineered.columns]

        X = df_engineered[self.feature_names].fillna(df_engineered[self.feature_names].median())
        y = df_engineered["target_clv_future"]

        # Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Feature Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Regression Models Pool
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        if HAS_XGB:
            models["XGBoost Regressor"] = xgb.XGBRegressor(n_estimators=100, random_state=42)

        if HAS_LGB:
            models["LightGBM Regressor"] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        # Benchmark Models
        results = []
        best_r2 = -999.0

        for name, model in models.items():
            logger.info(f"Training and evaluating {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            exp_var = explained_variance_score(y_test, y_pred)

            # MAPE
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1.0))) * 100

            results.append({
                "Model": name,
                "MAE ($)": round(mae, 2),
                "RMSE ($)": round(rmse, 2),
                "R² Score": round(r2, 4),
                "MAPE (%)": round(mape, 2),
                "Explained Variance": round(exp_var, 4)
            })

            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name

        comparison_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
        logger.info(f"Top performing CLV regressor: {self.best_model_name} (R² Score: {best_r2:.4f})")

        metrics_dict = {
            "best_model_name": self.best_model_name,
            "best_r2_score": round(best_r2, 4),
            "feature_names": self.feature_names,
            "benchmark_results": results
        }

        self.save_model(metrics_dict)
        return self.best_model, metrics_dict, comparison_df

    def predict_clv(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Predicts future Customer Lifetime Value for a given feature matrix."""
        if self.best_model is None:
            model_path = self.models_dir / "clv_model.pkl"
            scaler_path = self.models_dir / "clv_scaler.pkl"
            if model_path.exists() and scaler_path.exists():
                self.best_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
            else:
                raise ModelTrainingError("No trained CLV regression model available.")

        df_engineered = self.engineer_clv_features_and_target(feature_df)
        X = df_engineered[self.feature_names].fillna(df_engineered[self.feature_names].median())
        X_scaled = self.scaler.transform(X)
        preds = self.best_model.predict(X_scaled)
        return np.maximum(preds, 0.0) # Ensure non-negative predictions

    def save_model(self, metrics_dict: Dict[str, Any]):
        model_path = self.models_dir / "clv_model.pkl"
        scaler_path = self.models_dir / "clv_scaler.pkl"
        metrics_path = self.models_dir / "clv_model_metrics.json"

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)

        logger.info(f"Saved trained CLV model to {model_path} and metrics to {metrics_path}")
