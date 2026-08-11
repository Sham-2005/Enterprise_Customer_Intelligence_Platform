"""
FastAPI Routes for MLOps, Model Registry, Drift Auditing & AI Governance Services.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas.mlops_schemas import RollbackRequest, RetrainRequest, SystemHealthResponse
from backend.mlops.registry import ModelRegistry
from backend.mlops.experiment_tracker import ExperimentTracker
from backend.mlops.monitoring import PerformanceMonitor
from backend.mlops.retraining import RetrainingPipeline
from config.settings import Settings

router = APIRouter(prefix="/api/v1/mlops", tags=["Enterprise MLOps & Governance"])

settings = Settings()
registry = ModelRegistry()
tracker = ExperimentTracker()
monitor = PerformanceMonitor()

@router.get("/health", response_model=SystemHealthResponse)
def mlops_health_check():
    return SystemHealthResponse(**monitor.get_system_health())

@router.get("/registry")
def get_model_registry():
    return registry.get_registered_models()

@router.post("/rollback")
def rollback_model_version(request: RollbackRequest):
    success = registry.rollback_version(request.model_name, request.target_version)
    if not success:
        raise HTTPException(status_code=400, detail=f"Rollback failed for {request.model_name} to {request.target_version}")
    return {"status": "success", "model": request.model_name, "active_version": request.target_version}

@router.get("/experiments")
def get_experiments():
    return tracker.get_experiments()

@router.get("/drift")
def get_drift_report():
    reports_dir = settings.get_path("paths.reports_dir")
    json_path = reports_dir / "drift_report.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Drift report missing. Run MLOps pipeline first.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@router.post("/retrain")
def trigger_retraining(request: RetrainRequest):
    feature_store_path = settings.get_path("paths.output_dir") / "feature_store.csv"
    if not feature_store_path.exists():
        raise HTTPException(status_code=404, detail="Feature store missing.")
    df = pd.read_csv(feature_store_path)
    
    retrainer = RetrainingPipeline()
    res = retrainer.retrain_churn_model(df, request.new_version)
    return {"status": "success", "new_model_entry": res}
