"""
Centralized Model Registry & Version Control System for ECIP.
Tracks model lifecycle states, metrics, features, hyperparameters, and version rollbacks.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.ModelRegistry")

class ModelRegistry:
    """Enterprise Model Registry managing lifecycle stages, versioning, and governance."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.models_dir = self.settings.get_path("paths.models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "model_registry.json"
        self._registry: Dict[str, Any] = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse registry file: {e}")
        return {"registered_models": {}, "last_updated": datetime.now().isoformat()}

    def register_model(
        self,
        model_name: str,
        version: str,
        algorithm: str,
        metrics: Dict[str, float],
        features: List[str],
        hyperparameters: Dict[str, Any],
        owner: str = "ECIP-MLOps-Team",
        status: str = "Active"
    ) -> Dict[str, Any]:
        """Registers a model version in the central registry."""
        entry = {
            "model_name": model_name,
            "version": version,
            "algorithm": algorithm,
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "features": features,
            "hyperparameters": hyperparameters,
            "status": status,
            "owner": owner
        }

        if model_name not in self._registry["registered_models"]:
            self._registry["registered_models"][model_name] = {
                "active_version": version,
                "versions": {}
            }

        self._registry["registered_models"][model_name]["versions"][version] = entry
        if status == "Active":
            self._registry["registered_models"][model_name]["active_version"] = version

        self._save_registry()
        logger.info(f"Registered model '{model_name}' version '{version}' with status '{status}'.")
        return entry

    def rollback_version(self, model_name: str, target_version: str) -> bool:
        """Rolls back the active model pointer to a previous version."""
        if model_name in self._registry["registered_models"]:
            versions = self._registry["registered_models"][model_name]["versions"]
            if target_version in versions:
                self._registry["registered_models"][model_name]["active_version"] = target_version
                for v, data in versions.items():
                    data["status"] = "Active" if v == target_version else "Archived"
                self._save_registry()
                logger.info(f"Rolled back model '{model_name}' to version '{target_version}'.")
                return True
        return False

    def get_registered_models(self) -> Dict[str, Any]:
        return self._registry.get("registered_models", {})

    def _save_registry(self):
        self._registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=4)
