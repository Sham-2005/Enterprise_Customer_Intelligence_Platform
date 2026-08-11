"""
Pydantic DTO Schemas for ECIP MLOps & Governance REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class RollbackRequest(BaseModel):
    model_name: str = Field(..., example="ChurnClassifier")
    target_version: str = Field(..., example="v1.0")

class RetrainRequest(BaseModel):
    model_name: str = Field(..., example="ChurnClassifier")
    new_version: str = Field(..., example="v2.0")

class SystemHealthResponse(BaseModel):
    status: str
    total_requests: int
    average_latency_ms: float
    system_uptime: str
