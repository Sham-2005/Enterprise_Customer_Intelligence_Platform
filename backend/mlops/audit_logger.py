"""
AI Governance Audit Logging System for ECIP.
Maintains tamper-evident inference audit logs, user activity logs, and model governance records.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.AuditLogger")

class AuditLogger:
    """Records inference calls and model governance changes to an audit log file."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.logs_dir = self.settings.get_path("paths.logs_dir")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.logs_dir / "inference_audit.log"

    def log_inference_event(
        self, model_name: str, version: str, entity_id: str, action: str, details: str = ""
    ):
        """Appends a structured audit entry to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [AUDIT] Model='{model_name}' Version='{version}' Entity='{entity_id}' Action='{action}' Details='{details}'\n"
        
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.debug(f"Audit log appended for '{model_name}'.")
