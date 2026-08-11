"""
FastAPI Routes for AI Recommendation Engine & Personalization Services.
"""

from typing import Dict, Any, List
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas.recommendation_schemas import (
    CustomerRecommendationRequest, CustomerRecommendationResponse,
    SimilarProductsRequest
)
from backend.models.recommender import HybridRecommenderEngine
from config.settings import Settings

router = APIRouter(prefix="/api/v1/recommendations", tags=["AI Recommendation Engine"])

settings = Settings()
master_path = settings.get_path("paths.output_dir") / "master_dataset.csv"

# Shared Recommender Engine
_recommender_engine = None

def get_recommender():
    global _recommender_engine
    if _recommender_engine is None:
        if not master_path.exists():
            raise HTTPException(status_code=404, detail="Master dataset missing. Run pipeline first.")
        master_df = pd.read_csv(master_path)
        _recommender_engine = HybridRecommenderEngine()
        _recommender_engine.fit(master_df)
    return _recommender_engine

@router.get("/health")
def recommendation_health_check():
    return {"status": "healthy", "service": "ECIP Recommendation Engine API", "version": "1.0.0"}

@router.get("/metrics")
def get_recommendation_metrics():
    engine = get_recommender()
    return engine.evaluate_metrics()

@router.post("/customer", response_model=CustomerRecommendationResponse)
def get_customer_recommendations(request: CustomerRecommendationRequest):
    try:
        engine = get_recommender()
        recs = engine.recommend_for_customer(request.customer_id, request.top_n)
        return CustomerRecommendationResponse(
            customer_id=request.customer_id,
            recommendations=recs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similar-products")
def get_similar_products(request: SimilarProductsRequest):
    try:
        engine = get_recommender()
        return engine.get_similar_products(request.product_id, request.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
