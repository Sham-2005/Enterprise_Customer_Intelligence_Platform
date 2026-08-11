"""
Execution Script for ECIP Market Basket Analysis & Association Rule Mining Pipeline.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings
from backend.analytics.market_basket import MarketBasketAnalyzer
from utils.logger import setup_logger

logger = setup_logger("ECIP.RunMBA")

def main():
    try:
        logger.info("=" * 60)
        logger.info("STARTING MARKET BASKET ANALYSIS & ASSOCIATION RULE PIPELINE")
        logger.info("=" * 60)

        settings = Settings()
        output_dir = settings.get_path("paths.output_dir")
        models_dir = settings.get_path("paths.models_dir")
        models_dir.mkdir(parents=True, exist_ok=True)

        master_path = output_dir / "master_dataset.csv"
        if not master_path.exists():
            logger.error("Master dataset CSV missing! Please execute `python run_pipeline.py` first.")
            sys.exit(1)

        master_df = pd.read_csv(master_path)
        logger.info(f"Loaded Master Dataset with shape: {master_df.shape}")

        # Step 1: Run Market Basket Mining Engine
        analyzer = MarketBasketAnalyzer()
        rules_df, bundles_df, cross_sell_df, metrics_dict = analyzer.analyze_market_basket(master_df)

        # Step 2: Save Data Warehouse Persistence Artifacts
        rules_path = output_dir / "association_rules.csv"
        rules_df.to_csv(rules_path, index=False)
        logger.info(f"Exported Association Rules to {rules_path}")

        bundles_path = output_dir / "product_bundles.csv"
        bundles_df.to_csv(bundles_path, index=False)
        logger.info(f"Exported Product Bundles to {bundles_path}")

        cross_path = output_dir / "cross_sell_recommendations.csv"
        cross_sell_df.to_csv(cross_path, index=False)
        logger.info(f"Exported Cross-Sell Recommendations to {cross_path}")

        basket_stats_path = output_dir / "basket_statistics.csv"
        basket_stats_df = pd.DataFrame([metrics_dict])
        basket_stats_df.to_csv(basket_stats_path, index=False)
        logger.info(f"Exported Basket Statistics to {basket_stats_path}")

        metrics_path = models_dir / "mba_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Exported MBA Metrics to {metrics_path}")

        logger.info("=" * 60)
        logger.info("MARKET BASKET ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Market Basket Analysis pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
