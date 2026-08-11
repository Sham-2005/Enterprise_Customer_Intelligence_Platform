"""
Unit Tests for Raw Dataset Ingestion and File Integrity.
"""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.data.loader import DataLoader

def test_file_existence_check():
    loader = DataLoader("config/config.yaml")
    existence = loader.check_file_existence()
    assert isinstance(existence, dict)
    assert "customers" in existence
    assert "orders" in existence
