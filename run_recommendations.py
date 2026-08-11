"""
Execution Script for ECIP AI Recommendation Engine & Personalization Pipeline.
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
from backend.models.recommender import HybridRecommenderEngine
from utils.logger import setup_logger

logger = setup_logger("ECIP.RunRecommendations")

def main():
    try:
        logger.info("=" * 60)
        logger.info("STARTING AI RECOMMENDATION & PERSONALIZATION PIPELINE")
        logger.info("=" * 60)

        settings = Settings()
        output_dir = settings.get_path("paths.output_dir")
        models_dir = settings.get_path("paths.models_dir")
        models_dir.mkdir(parents=True, exist_ok=True)

        master_path = output_dir / "master_dataset.csv"
        feature_store_path = output_dir / "feature_store.csv"

        if not master_path.exists() or not feature_store_path.exists():
            logger.error("Master dataset or Feature Store CSV missing! Run `python run_pipeline.py` first.")
            sys.exit(1)

        master_df = pd.read_csv(master_path)
        feature_store_df = pd.read_csv(feature_store_path)
        logger.info(f"Loaded Master Dataset ({len(master_df):,} rows) and Feature Store ({len(feature_store_df):,} customers).")

        # Step 1: Fit Hybrid Engine
        logger.info("Fitting Collaborative & Content-Based Hybrid Recommender...")
        recommender = HybridRecommenderEngine()
        recommender.fit(master_df)

        # Step 2: Generate Batch Customer Recommendations
        logger.info("Scoring personalized product recommendations for customer sample...")
        sample_customers = feature_store_df["customer_unique_id"].head(500).tolist()
        rec_rows = []

        for cid in sample_customers:
            recs = recommender.recommend_for_customer(cid, top_n=5)
            for r in recs:
                rec_rows.append({
                    "customer_unique_id": cid,
                    "recommended_product_id": r["product_id"],
                    "category": r["category"],
                    "avg_price": r["avg_price"],
                    "hybrid_score": r["hybrid_score"],
                    "explanation": r["explanation"]
                })

        customer_recs_df = pd.DataFrame(rec_rows)

        # Step 3: Trending Products & Cross-Sell Datasets
        trending_df = recommender.top_trending_products
        metrics = recommender.evaluate_metrics()

        # Step 4: Export Persistence
        rec_prod_path = output_dir / "recommended_products.csv"
        customer_recs_df.to_csv(rec_prod_path, index=False)
        logger.info(f"Exported Recommended Products to {rec_prod_path}")

        cust_rec_path = output_dir / "customer_recommendations.csv"
        customer_recs_df.to_csv(cust_rec_path, index=False)
        logger.info(f"Exported Customer Recommendations to {cust_rec_path}")

        trending_path = output_dir / "trending_products.csv"
        trending_df.to_csv(trending_path, index=False)
        logger.info(f"Exported Trending Products to {trending_path}")

        cross_path = output_dir / "cross_sell_products.csv"
        trending_df.head(50).to_csv(cross_path, index=False)
        logger.info(f"Exported Cross-Sell Products to {cross_path}")

        upsell_path = output_dir / "upsell_products.csv"
        trending_df.sort_values(by="avg_price", ascending=False).head(50).to_csv(upsell_path, index=False)
        logger.info(f"Exported Upsell Products to {upsell_path}")

        metrics_path = models_dir / "recommendation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Exported Recommendation Metrics to {metrics_path}")

        logger.info("=" * 60)
        logger.info("AI RECOMMENDATION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Recommendation pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
