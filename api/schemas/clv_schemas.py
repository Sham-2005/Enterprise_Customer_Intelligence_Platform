"""
Pydantic DTO Schemas for ECIP Customer Lifetime Value (CLV) REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class SingleCLVPredictionRequest(BaseModel):
    total_spending: float = Field(..., example=450.00)
    total_orders: int = Field(..., example=4)
    avg_order_value: float = Field(..., example=112.50)
    recency_days: float = Field(..., example=25.0)
    historical_clv: float = Field(..., example=450.00)
    avg_review_score_given: float = Field(..., example=4.5)
    distinct_categories_count: int = Field(..., example=3)
    loyalty_score: float = Field(..., example=78.0)

class SingleCLVPredictionResponse(BaseModel):
    customer_id: Optional[str] = "single_customer"
    predicted_clv_12m: float
    clv_value_tier: str
    revenue_opportunity_type: str
    revenue_recommendation: str
    natural_language_explanation: str

class CLVModelInfoResponse(BaseModel):
    model_name: str
    r2_score: float
    feature_names: List[str]
    benchmark_results: List[Dict[str, Any]]
