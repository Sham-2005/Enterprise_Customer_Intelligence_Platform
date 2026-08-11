"""
Pydantic DTO Schemas for ECIP Recommendation Engine REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class CustomerRecommendationRequest(BaseModel):
    customer_id: str = Field(..., example="00012a254244fe58809e000e32ac82a0")
    top_n: Optional[int] = 5

class ProductRecommendationItem(BaseModel):
    product_id: str
    category: str
    avg_price: float
    hybrid_score: float
    explanation: str

class CustomerRecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[ProductRecommendationItem]

class SimilarProductsRequest(BaseModel):
    product_id: str = Field(..., example="8c8d80344217396f632301c360829828")
    top_n: Optional[int] = 5
