"""
Experiment Tracking System for ECIP.
Records training runs, hyperparameters, metrics, hardware resource notes, and experiment history.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.ExperimentTracker")

class ExperimentTracker:
    """Tracks machine learning experiment runs and hyperparameter tuning diagnostics."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.models_dir = self.settings.get_path("paths.models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.models_dir / "experiments_log.json"
        self.experiments: List[Dict[str, Any]] = self._load_experiments()

    def _load_experiments(self) -> List[Dict[str, Any]]:
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read experiment log file: {e}")
        return []

    def log_experiment(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        duration_seconds: float,
        dataset_version: str = "v1.0",
        notes: str = "Automated baseline run"
    ) -> Dict[str, Any]:
        """Logs an experiment run entry to disk."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        entry = {
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "dataset_version": dataset_version,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "duration_seconds": round(duration_seconds, 2),
            "notes": notes
        }

        self.experiments.append(entry)
        self._save_experiments()
        logger.info(f"Logged experiment run '{run_id}' for model '{model_name}'.")
        return entry

    def get_experiments(self) -> List[Dict[str, Any]]:
        return self.experiments

    def _save_experiments(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=4)
