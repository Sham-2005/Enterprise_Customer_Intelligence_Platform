"""
Unit Tests for Machine Learning Risk Engines and Stratification.
"""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.models.risk_engine import ChurnRiskEngine

def test_risk_stratification():
    risk_engine = ChurnRiskEngine()
    assert risk_engine.stratify_risk(0.85) == "Critical"
    assert risk_engine.stratify_risk(0.65) == "High"
    assert risk_engine.stratify_risk(0.45) == "Medium"
    assert risk_engine.stratify_risk(0.25) == "Low"
    assert risk_engine.stratify_risk(0.10) == "Very Low"
