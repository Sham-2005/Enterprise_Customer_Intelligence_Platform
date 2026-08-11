"""
Pydantic DTO Schemas for ECIP Market Basket Analysis REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AssociationRuleItem(BaseModel):
    antecedents_str: str
    consequents_str: str
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float

class ProductBundleItem(BaseModel):
    bundle_name: str
    primary_category: str
    addon_category: str
    lift_score: float
    confidence_pct: float
    estimated_bundle_price: float
    projected_revenue_potential: float
    merchandising_strategy: str

class MBAMetricsResponse(BaseModel):
    total_transactions: int
    unique_categories: int
    frequent_itemsets_count: int
    association_rules_count: int
    avg_lift: float
    avg_confidence: float
