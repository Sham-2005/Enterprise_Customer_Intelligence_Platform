"""
FastAPI Routes for CLV Inference & Revenue Intelligence Services.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas.clv_schemas import (
    SingleCLVPredictionRequest, SingleCLVPredictionResponse, CLVModelInfoResponse
)
from backend.models.clv_model import CLVModelPipeline
from backend.models.revenue_engine import RevenueIntelligenceEngine
from backend.explainability.clv_explainer import CLVSHAPExplainer
from config.settings import Settings

router = APIRouter(prefix="/api/v1/clv", tags=["Customer Lifetime Value & Revenue Intelligence"])

settings = Settings()
models_dir = settings.get_path("paths.models_dir")
metrics_file = models_dir / "clv_model_metrics.json"

@router.get("/health")
def clv_health_check():
    return {"status": "healthy", "service": "ECIP CLV & Revenue Intelligence API", "version": "1.0.0"}

@router.get("/model-info", response_model=CLVModelInfoResponse)
def get_clv_model_info():
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="CLV metrics file not found. Please train model first.")

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return CLVModelInfoResponse(
        model_name=data.get("best_model_name", "Unknown"),
        r2_score=data.get("best_r2_score", 0.0),
        feature_names=data.get("feature_names", []),
        benchmark_results=data.get("benchmark_results", [])
    )

@router.post("/predict", response_model=SingleCLVPredictionResponse)
def predict_single_clv(request: SingleCLVPredictionRequest):
    try:
        pipeline = CLVModelPipeline()
        revenue_engine = RevenueIntelligenceEngine()

        df_input = pd.DataFrame([request.model_dump()])

        pred_clv = float(pipeline.predict_clv(df_input)[0])
        df_res, _, _ = revenue_engine.generate_opportunity_blueprints(df_input, [pred_clv])

        explainer = CLVSHAPExplainer(pipeline.best_model, pipeline.feature_names)
        shap_res = explainer.explain_clv_instance(df_input)

        row = df_res.iloc[0]
        return SingleCLVPredictionResponse(
            predicted_clv_12m=round(pred_clv, 2),
            clv_value_tier=str(row["clv_value_tier"]),
            revenue_opportunity_type=str(row["revenue_opportunity_type"]),
            revenue_recommendation=str(row["revenue_recommendation"]),
            natural_language_explanation=shap_res["natural_language_explanation"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
