"""
Unit Tests for Configuration Manager and Settings.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings

def test_settings_load():
    settings = Settings("config/config.yaml")
    assert settings.config is not None
    assert settings.get("app.name") == "Enterprise Customer Intelligence Platform"
    assert settings.get("api.port") == 8000
