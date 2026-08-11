"""
FastAPI Routes for Churn Inference & Risk Engine Services.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from api.schemas.churn_schemas import (
    SinglePredictionRequest, SinglePredictionResponse, ModelInfoResponse
)
from backend.models.churn_model import ChurnModelPipeline
from backend.models.risk_engine import ChurnRiskEngine
from backend.explainability.explainer import SHAPExplainer
from config.settings import Settings

router = APIRouter(prefix="/api/v1/churn", tags=["Customer Churn Intelligence"])

# Global Model & Engine Instances
settings = Settings()
models_dir = settings.get_path("paths.models_dir")
metrics_file = models_dir / "model_metrics.json"

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "ECIP Churn Risk API", "version": "1.0.0"}

@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Model metrics file not found. Please train model first.")

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ModelInfoResponse(
        model_name=data.get("best_model_name", "Unknown"),
        roc_auc_score=data.get("best_roc_auc", 0.0),
        feature_names=data.get("feature_names", []),
        metrics_summary=data.get("benchmark_results", [])
    )

@router.post("/predict", response_model=SinglePredictionResponse)
def predict_single_customer(request: SinglePredictionRequest):
    try:
        pipeline = ChurnModelPipeline()
        risk_engine = ChurnRiskEngine()

        df_input = pd.DataFrame([request.model_dump()])

        prob = float(pipeline.predict_churn_probability(df_input)[0])
        risk_level = risk_engine.stratify_risk(prob)

        # XAI Explainer
        explainer = SHAPExplainer(pipeline.best_model, pipeline.feature_names)
        shap_res = explainer.explain_instance(df_input)

        # Actions
        _, high_risk_df = risk_engine.generate_retention_recommendations(df_input, [prob])
        actions_str = high_risk_df["recommended_retention_actions"].iloc[0] if not high_risk_df.empty else "Maintain standard engagement."
        action_plan = high_risk_df["retention_action_plan"].iloc[0] if not high_risk_df.empty else "Standard flow."

        return SinglePredictionResponse(
            churn_probability=round(prob, 4),
            risk_level=risk_level,
            natural_language_explanation=shap_res["natural_language_explanation"],
            top_risk_factors=shap_res["top_risk_factors"],
            recommended_retention_actions=actions_str.split(" | "),
            retention_action_plan=action_plan
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
