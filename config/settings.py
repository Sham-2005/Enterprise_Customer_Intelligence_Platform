"""
Configuration loader and settings validator for ECIP.
Reads YAML configurations dynamically without hardcoding paths.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict
import importlib.util

try:
    import yaml
except ImportError:
    yaml = None

try:
    from utils.exceptions import ConfigurationError
except (ImportError, AttributeError):
    _exc_spec = importlib.util.spec_from_file_location("root_utils_exceptions", project_root / "utils" / "exceptions.py")
    _exc_mod = importlib.util.module_from_spec(_exc_spec)
    _exc_spec.loader.exec_module(_exc_mod)
    ConfigurationError = _exc_mod.ConfigurationError

try:
    from utils.logger import setup_logger
except (ImportError, AttributeError):
    _log_spec = importlib.util.spec_from_file_location("root_utils_logger", project_root / "utils" / "logger.py")
    _log_mod = importlib.util.module_from_spec(_log_spec)
    _log_spec.loader.exec_module(_log_mod)
    setup_logger = _log_mod.setup_logger

logger = setup_logger("ECIP.Config")

class Settings:
    """Singleton Configuration Manager for ECIP."""
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = "config/config.yaml"):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            # Fallback relative to project root
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / config_path

        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found at {config_path}",
                details=f"Absolute path checked: {path.absolute()}"
            )

        try:
            if yaml is not None:
                with open(path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f)
            else:
                # Fallback default dictionary configuration
                self._config = {
                    "system": {"name": "ECIP", "environment": "development"},
                    "paths": {"dataset_dir": "data_set", "output_dir": "output", "models_dir": "output/models", "reports_dir": "output/reports", "logs_dir": "output/logs"},
                    "dataset_files": {
                        "customers": "olist_customers_dataset.csv",
                        "orders": "olist_orders_dataset.csv",
                        "order_items": "olist_order_items_dataset.csv",
                        "products": "olist_products_dataset.csv",
                        "payments": "olist_order_payments_dataset.csv",
                        "reviews": "olist_order_reviews_dataset.csv",
                        "sellers": "olist_sellers_dataset.csv",
                        "geolocation": "olist_geolocation_dataset.csv",
                        "category_translation": "product_category_name_translation.csv"
                    }
                }
            logger.info(f"Loaded configuration successfully from {path}")
        except Exception as e:
            raise ConfigurationError(
                "Failed to parse YAML configuration", details=str(e)
            )

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """Fetch nested dictionary values using dot notation (e.g. 'paths.dataset_dir')."""
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def get_path(self, key_path: str) -> Path:
        """Resolves path string relative to project root directory."""
        path_str = self.get(key_path)
        if not path_str:
            raise ConfigurationError(f"Configuration key '{key_path}' not defined.")
        
        project_root = Path(__file__).resolve().parent.parent
        resolved = project_root / path_str
        return resolved
