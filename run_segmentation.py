"""
Execution Script for ECIP AI Customer Segmentation & RFM Intelligence Pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings
from backend.analytics.rfm import RFMAnalyzer
from backend.analytics.segmentation import CustomerSegmentationEngine
from backend.analytics.persona_generator import PersonaGenerator
from utils.logger import setup_logger

logger = setup_logger("ECIP.RunSegmentation")

def main():
    try:
        logger.info("=" * 60)
        logger.info("STARTING AI CUSTOMER SEGMENTATION & RFM PIPELINE")
        logger.info("=" * 60)

        settings = Settings()
        output_dir = settings.get_path("paths.output_dir")
        feature_store_path = output_dir / "feature_store.csv"

        if not feature_store_path.exists():
            logger.error("Feature store CSV missing! Please execute `python run_pipeline.py` first.")
            sys.exit(1)

        feature_store_df = pd.read_csv(feature_store_path)
        logger.info(f"Loaded Feature Store dataset with shape: {feature_store_df.shape}")

        # Step 1: RFM Intelligence Analysis
        logger.info("Running RFM Intelligence Engine...")
        rfm_analyzer = RFMAnalyzer()
        rfm_df, rfm_summary = rfm_analyzer.analyze_rfm(feature_store_df)

        rfm_scores_path = output_dir / "rfm_scores.csv"
        rfm_df.to_csv(rfm_scores_path, index=False)
        logger.info(f"Exported RFM Scores to {rfm_scores_path}")

        # Step 2: Unsupervised ML Clustering & Model Comparison
        logger.info("Running AI Unsupervised Clustering Engine...")
        seg_engine = CustomerSegmentationEngine()
        segmented_df, benchmark_df, meta = seg_engine.run_segmentation(feature_store_df)

        benchmark_path = output_dir / "model_benchmark.csv"
        benchmark_df.to_csv(benchmark_path, index=False)
        logger.info(f"Exported Clustering Algorithm Benchmark to {benchmark_path}")

        # Step 3: Business Persona Generation
        logger.info("Generating Business Personas & Recommendations...")
        persona_gen = PersonaGenerator()
        segmented_df, personas_df = persona_gen.generate_personas(segmented_df)

        segments_path = output_dir / "customer_segments.csv"
        segmented_df.to_csv(segments_path, index=False)
        logger.info(f"Exported Customer Segments to {segments_path}")

        personas_path = output_dir / "customer_personas.csv"
        personas_df.to_csv(personas_path, index=False)
        logger.info(f"Exported Business Personas to {personas_path}")

        # Step 4: Save Summary Report
        report_path = settings.get_path("paths.reports_dir") / "segmentation_report.csv"
        personas_df.to_csv(report_path, index=False)
        logger.info(f"Exported Segmentation Summary Report to {report_path}")

        logger.info("=" * 60)
        logger.info("AI CUSTOMER SEGMENTATION PIPELINE FINISHED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Segmentation pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
