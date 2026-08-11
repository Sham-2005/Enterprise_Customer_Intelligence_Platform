"""
Entrypoint script for executing ECIP Data Engineering & Feature Engineering Pipeline.
"""

import sys
from pathlib import Path

# Ensure project root is in python module search path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.data.pipeline import ETLPipeline
from utils.logger import setup_logger

logger = setup_logger("ECIP.Main")

def main():
    try:
        pipeline = ETLPipeline(config_path="config/config.yaml")
        output_paths = pipeline.run()
        
        logger.info("Pipeline execution summary:")
        for name, path in output_paths.items():
            logger.info(f" - {name}: {path.name} ({path.stat().st_size / (1024*1024):.2f} MB)")

    except Exception as e:
        logger.error(f"ETL Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
