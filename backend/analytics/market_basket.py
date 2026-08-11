"""
Enterprise Market Basket Analysis & Association Rule Mining Engine for ECIP.
Implements transaction basket encoding, Apriori and FP-Growth frequent itemset mining,
association rule generation (Support, Confidence, Lift, Leverage, Conviction),
product bundle identification, and merchandising strategy recommendation.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
except ImportError:
    apriori = None
    fpgrowth = None
    association_rules = None


from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.MarketBasketAnalyzer")

class MarketBasketAnalyzer:
    """Engineers transaction baskets, mines frequent itemsets, and generates association rules."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.min_support = self.settings.get("analytics.market_basket.min_support", 0.005)
        self.min_threshold = self.settings.get("analytics.market_basket.min_threshold", 1.0)

    def analyze_market_basket(
        self, master_df: pd.DataFrame, use_fpgrowth: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes end-to-end transaction encoding, rule mining, bundle generation, and metric reporting.

        Returns:
            Tuple[rules_df, bundles_df, cross_sell_df, metrics_dict]
        """
        logger.info("Starting Market Basket Analysis & Association Rule Mining...")

        # Step 1: Transaction Encoding (Order ID vs Category)
        logger.info("Constructing transaction basket one-hot matrix...")
        basket = master_df.groupby(["order_id", "product_category_name_english"])["order_item_id"].count().unstack().fillna(0)
        basket_encoded = basket.map(lambda x: 1 if x > 0 else 0)

        logger.info(f"Transaction matrix created: {basket_encoded.shape[0]} orders, {basket_encoded.shape[1]} categories.")

        # Step 2: Mine Frequent Itemsets (FP-Growth / Apriori)
        if use_fpgrowth:
            logger.info(f"Mining frequent itemsets using FP-Growth (min_support={self.min_support})...")
            frequent_itemsets = fpgrowth(basket_encoded, min_support=self.min_support, use_colnames=True)
        else:
            logger.info(f"Mining frequent itemsets using Apriori (min_support={self.min_support})...")
            frequent_itemsets = apriori(basket_encoded, min_support=self.min_support, use_colnames=True)

        if frequent_itemsets.empty:
            logger.warning("No frequent itemsets found with current min_support threshold. Retrying with 0.001...")
            frequent_itemsets = fpgrowth(basket_encoded, min_support=0.001, use_colnames=True)

        # Step 3: Generate Association Rules
        logger.info("Generating association rules (metric='lift')...")
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values(by="lift", ascending=False)
        else:
            rules = pd.DataFrame()

        # Step 4: Clean Rules and Extract Bundles
        rules_df, bundles_df, cross_sell_df = self._process_rules_and_bundles(rules, master_df)

        metrics_dict = {
            "total_transactions": len(basket_encoded),
            "unique_categories": len(basket_encoded.columns),
            "frequent_itemsets_count": len(frequent_itemsets),
            "association_rules_count": len(rules_df),
            "avg_lift": round(float(rules_df["lift"].mean()), 4) if not rules_df.empty else 0.0,
            "avg_confidence": round(float(rules_df["confidence"].mean()), 4) if not rules_df.empty else 0.0
        }

        logger.info(f"Market Basket Analysis completed: {len(rules_df)} association rules generated.")
        return rules_df, bundles_df, cross_sell_df, metrics_dict

    def _process_rules_and_bundles(
        self, rules: pd.DataFrame, master_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if rules.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = rules.copy()
        df["antecedents_str"] = df["antecedents"].apply(lambda x: ", ".join(list(x)))
        df["consequents_str"] = df["consequents"].apply(lambda x: ", ".join(list(x)))

        # Product Bundles & Merchandising Strategies
        bundles = []
        cross_sell = []

        for _, row in df.head(30).iterrows():
            ant = row["antecedents_str"]
            cons = row["consequents_str"]
            lift = row["lift"]
            conf = row["confidence"]
            supp = row["support"]

            # Estimate bundle revenue potential
            cat_spend = master_df[master_df["product_category_name_english"].isin([ant, cons])]["price"].mean() * 1.8
            proj_revenue = cat_spend * (supp * len(master_df["order_id"].unique()))

            bundles.append({
                "bundle_name": f"{ant} + {cons}",
                "primary_category": ant,
                "addon_category": cons,
                "lift_score": round(lift, 2),
                "confidence_pct": round(conf * 100, 1),
                "support_pct": round(supp * 100, 2),
                "estimated_bundle_price": round(cat_spend, 2),
                "projected_revenue_potential": round(proj_revenue, 2),
                "merchandising_strategy": f"Promote joint '{ant} & {cons}' bundle with 10% discount on {cons}."
            })

            cross_sell.append({
                "trigger_category": ant,
                "recommended_cross_sell": cons,
                "conversion_probability": round(conf * 100, 1),
                "marketing_channel": "Checkout Pop-up / Post-Purchase Email",
                "promotional_headline": f"Add {cons} to your order and save!"
            })

        bundles_df = pd.DataFrame(bundles)
        cross_sell_df = pd.DataFrame(cross_sell)

        return df, bundles_df, cross_sell_df
