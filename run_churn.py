"""
Execution Script for ECIP AI Customer Churn Prediction & Risk Intelligence Pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings
from backend.models.churn_model import ChurnModelPipeline
from backend.models.risk_engine import ChurnRiskEngine
from backend.explainability.explainer import SHAPExplainer
from utils.logger import setup_logger

logger = setup_logger("ECIP.RunChurn")

def main():
    try:
        logger.info("=" * 60)
        logger.info("STARTING AI CUSTOMER CHURN PREDICTION & RISK PIPELINE")
        logger.info("=" * 60)

        settings = Settings()
        output_dir = settings.get_path("paths.output_dir")
        feature_store_path = output_dir / "feature_store.csv"

        if not feature_store_path.exists():
            logger.error("Feature store CSV missing! Please execute `python run_pipeline.py` first.")
            sys.exit(1)

        feature_store_df = pd.read_csv(feature_store_path)
        logger.info(f"Loaded Feature Store dataset with shape: {feature_store_df.shape}")

        # Step 1: Model Training & Evaluation Benchmark
        logger.info("Training and evaluating classification models...")
        model_pipeline = ChurnModelPipeline()
        best_model, metrics_dict, comparison_df = model_pipeline.train_and_evaluate(feature_store_df)

        # Step 2: Inference & Risk Stratification
        logger.info("Scoring customer churn probabilities & stratifying risk levels...")
        probs = model_pipeline.predict_churn_probability(feature_store_df)
        
        risk_engine = ChurnRiskEngine()
        predictions_df, high_risk_df = risk_engine.generate_retention_recommendations(
            feature_store_df, probs
        )

        # Step 3: Explainable AI (SHAP)
        logger.info("Generating SHAP feature attributions & diagnostic reports...")
        explainer = SHAPExplainer(best_model, model_pipeline.feature_names)
        explainer.generate_shap_report(feature_store_df[model_pipeline.feature_names].head(200))

        # Step 4: Export Persistence
        pred_path = output_dir / "customer_churn_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Exported Churn Predictions to {pred_path}")

        high_risk_path = output_dir / "high_risk_customers.csv"
        high_risk_df.to_csv(high_risk_path, index=False)
        logger.info(f"Exported High Risk Roster ({len(high_risk_df):,} customers) to {high_risk_path}")

        recs_path = output_dir / "retention_recommendations.csv"
        rec_cols = ["customer_unique_id", "churn_probability", "risk_level", "recommended_retention_actions", "retention_action_plan"]
        avail_rec_cols = [c for c in rec_cols if c in predictions_df.columns]
        predictions_df[avail_rec_cols].to_csv(recs_path, index=False)
        logger.info(f"Exported Retention Recommendations to {recs_path}")

        logger.info("=" * 60)
        logger.info("AI CHURN PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Churn pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
