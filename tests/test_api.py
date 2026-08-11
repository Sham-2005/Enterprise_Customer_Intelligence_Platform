"""
Unit Tests for FastAPI REST Endpoints.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["platform"] == "Enterprise Customer Intelligence Platform (ECIP)"

def test_churn_health_endpoint():
    response = client.get("/api/v1/churn/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
