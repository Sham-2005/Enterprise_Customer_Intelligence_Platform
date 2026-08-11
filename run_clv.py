"""
Execution Script for ECIP AI Customer Lifetime Value (CLV) Prediction & Revenue Pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings
from backend.models.clv_model import CLVModelPipeline
from backend.models.revenue_engine import RevenueIntelligenceEngine
from backend.explainability.clv_explainer import CLVSHAPExplainer
from utils.logger import setup_logger

logger = setup_logger("ECIP.RunCLV")

def main():
    try:
        logger.info("=" * 60)
        logger.info("STARTING AI CUSTOMER LIFETIME VALUE (CLV) & REVENUE PIPELINE")
        logger.info("=" * 60)

        settings = Settings()
        output_dir = settings.get_path("paths.output_dir")
        feature_store_path = output_dir / "feature_store.csv"

        if not feature_store_path.exists():
            logger.error("Feature store CSV missing! Please execute `python run_pipeline.py` first.")
            sys.exit(1)

        feature_store_df = pd.read_csv(feature_store_path)
        logger.info(f"Loaded Feature Store dataset with shape: {feature_store_df.shape}")

        # Step 1: Model Training & Benchmark
        logger.info("Training and evaluating CLV regression models...")
        clv_pipeline = CLVModelPipeline()
        best_model, metrics_dict, comparison_df = clv_pipeline.train_and_evaluate(feature_store_df)

        # Step 2: Inference & Value Tier Stratification
        logger.info("Forecasting 12-month Customer Lifetime Value & engineering opportunity blueprints...")
        predicted_clv = clv_pipeline.predict_clv(feature_store_df)

        revenue_engine = RevenueIntelligenceEngine()
        predictions_df, high_value_df, forecast_df = revenue_engine.generate_opportunity_blueprints(
            feature_store_df, predicted_clv
        )

        # Step 3: SHAP Explainability
        logger.info("Fitting CLV SHAP explainer...")
        clv_df = clv_pipeline.engineer_clv_features_and_target(feature_store_df)
        explainer = CLVSHAPExplainer(best_model, clv_pipeline.feature_names)
        explainer.fit_explainer(clv_df[clv_pipeline.feature_names].head(100).values)

        # Step 4: Export Datasets
        pred_path = output_dir / "customer_clv_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Exported CLV Predictions to {pred_path}")

        value_seg_path = output_dir / "customer_value_segments.csv"
        predictions_df[["customer_unique_id", "predicted_clv", "clv_value_tier"]].to_csv(value_seg_path, index=False)
        logger.info(f"Exported Customer Value Segments to {value_seg_path}")

        forecast_path = output_dir / "revenue_forecast.csv"
        forecast_df.to_csv(forecast_path, index=False)
        logger.info(f"Exported Monthly Revenue Forecast to {forecast_path}")

        high_val_path = output_dir / "high_value_customers.csv"
        high_value_df.to_csv(high_val_path, index=False)
        logger.info(f"Exported High Value Customer Leaderboard ({len(high_value_df):,} customers) to {high_val_path}")

        dash_path = output_dir / "clv_dashboard_data.csv"
        predictions_df.to_csv(dash_path, index=False)
        logger.info(f"Exported CLV Dashboard Data to {dash_path}")

        logger.info("=" * 60)
        logger.info("AI CLV & REVENUE PIPELINE FINISHED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"CLV pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
