"""
Pydantic DTO Schemas for ECIP Churn REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class SinglePredictionRequest(BaseModel):
    total_spending: float = Field(..., example=250.50)
    total_orders: int = Field(..., example=3)
    avg_order_value: float = Field(..., example=83.50)
    recency_days: float = Field(..., example=120.0)
    historical_clv: float = Field(..., example=250.50)
    avg_review_score_given: float = Field(..., example=2.5)
    distinct_categories_count: int = Field(..., example=2)
    loyalty_score: float = Field(..., example=45.0)

class SinglePredictionResponse(BaseModel):
    customer_id: Optional[str] = "single_customer"
    churn_probability: float
    risk_level: str
    natural_language_explanation: str
    top_risk_factors: List[str]
    recommended_retention_actions: List[str]
    retention_action_plan: str

class ModelInfoResponse(BaseModel):
    model_name: str
    roc_auc_score: float
    feature_names: List[str]
    metrics_summary: List[Dict[str, Any]]
