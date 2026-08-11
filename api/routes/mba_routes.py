"""
FastAPI Routes for Market Basket Analysis & Association Rule Mining Services.
"""

from typing import Dict, Any, List
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas.mba_schemas import (
    AssociationRuleItem, ProductBundleItem, MBAMetricsResponse
)
from backend.analytics.market_basket import MarketBasketAnalyzer
from config.settings import Settings

router = APIRouter(prefix="/api/v1/mba", tags=["Market Basket Analysis"])

settings = Settings()
master_path = settings.get_path("paths.output_dir") / "master_dataset.csv"

_mba_cache = None

def get_mba_data():
    global _mba_cache
    if _mba_cache is None:
        if not master_path.exists():
            raise HTTPException(status_code=404, detail="Master dataset missing. Run pipeline first.")
        master_df = pd.read_csv(master_path)
        analyzer = MarketBasketAnalyzer()
        rules_df, bundles_df, cross_sell_df, metrics_dict = analyzer.analyze_market_basket(master_df)
        _mba_cache = (rules_df, bundles_df, cross_sell_df, metrics_dict)
    return _mba_cache

@router.get("/health")
def mba_health_check():
    return {"status": "healthy", "service": "ECIP Market Basket Analysis API", "version": "1.0.0"}

@router.get("/metrics", response_model=MBAMetricsResponse)
def get_mba_metrics():
    _, _, _, metrics = get_mba_data()
    return MBAMetricsResponse(**metrics)

@router.get("/rules")
def get_association_rules(min_lift: float = 1.0, limit: int = 20):
    rules_df, _, _, _ = get_mba_data()
    if rules_df.empty:
        return []
    filtered = rules_df[rules_df["lift"] >= min_lift].head(limit)
    return filtered[["antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction"]].to_dict(orient="records")

@router.get("/bundles")
def get_product_bundles(limit: int = 15):
    _, bundles_df, _, _ = get_mba_data()
    if bundles_df.empty:
        return []
    return bundles_df.head(limit).to_dict(orient="records")
