"""
Global Universal Search Engine for ECIP Executive Dashboard.
Allows searching across Customer ID, Order ID, Seller ID, and Product ID.
Returns structured entity metrics, profiles, and associated records.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("ECIP.SearchService")

class SearchService:
    """Universal entity lookup and cross-dataset search engine."""

    def search(
        self,
        query: str,
        master_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Executes search query across Customer, Order, Seller, and Product identifiers.

        Returns:
            Dict containing match_type, entity_id, summary_metrics, and matching_records.
        """
        query_str = str(query).strip()
        if not query_str or master_df.empty:
            return {
                "has_match": False,
                "query": query_str,
                "match_type": "None",
                "summary": {},
                "records": pd.DataFrame()
            }

        # 1. Check Customer ID Match
        cust_match = self._search_customer(query_str, master_df, feature_store_df)
        if cust_match["has_match"]:
            return cust_match

        # 2. Check Order ID Match
        order_match = self._search_order(query_str, master_df)
        if order_match["has_match"]:
            return order_match

        # 3. Check Seller ID Match
        seller_match = self._search_seller(query_str, master_df)
        if seller_match["has_match"]:
            return seller_match

        # 4. Check Product ID Match
        product_match = self._search_product(query_str, master_df)
        if product_match["has_match"]:
            return product_match

        # 5. Fallback Partial Match on Categories
        cat_match = self._search_category(query_str, master_df)
        if cat_match["has_match"]:
            return cat_match

        return {
            "has_match": False,
            "query": query_str,
            "match_type": "None",
            "summary": {"message": f"No matching Customer, Order, Seller, or Product found for '{query_str}'."},
            "records": pd.DataFrame()
        }

    def _search_customer(self, query: str, master_df: pd.DataFrame, feature_store_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        cust_col = None
        for c in ["customer_unique_id", "customer_id"]:
            if c in master_df.columns:
                matches = master_df[master_df[c].astype(str).str.contains(query, case=False, na=False)]
                if not matches.empty:
                    cust_id = matches[c].iloc[0]
                    records = master_df[master_df[c] == cust_id]
                    tot_spend = records["price"].sum() if "price" in records.columns else records["payment_value"].sum()
                    tot_orders = records["order_id"].nunique() if "order_id" in records.columns else len(records)
                    
                    fs_info = {}
                    if feature_store_df is not None and not feature_store_df.empty and "customer_unique_id" in feature_store_df.columns:
                        fs_cust = feature_store_df[feature_store_df["customer_unique_id"] == cust_id]
                        if not fs_cust.empty:
                            row = fs_cust.iloc[0]
                            fs_info = {
                                "churn_risk": row.get("risk_level", "Low Risk" if row.get("churn_label", 0) == 0 else "High Risk"),
                                "rfm_segment": row.get("rfm_segment", "N/A"),
                                "predicted_clv": f"${row.get('predicted_clv', tot_spend):,.2f}"
                            }

                    summary = {
                        "Entity Type": "Customer Profile",
                        "Customer ID": cust_id,
                        "Total Orders": f"{tot_orders:,}",
                        "Total Spending": f"${tot_spend:,.2f}",
                        "Avg Order Value": f"${tot_spend / max(tot_orders, 1):,.2f}",
                        **fs_info
                    }
                    return {"has_match": True, "query": query, "match_type": "Customer", "summary": summary, "records": records.head(50)}
        return {"has_match": False}

    def _search_order(self, query: str, master_df: pd.DataFrame) -> Dict[str, Any]:
        if "order_id" in master_df.columns:
            matches = master_df[master_df["order_id"].astype(str).str.contains(query, case=False, na=False)]
            if not matches.empty:
                ord_id = matches["order_id"].iloc[0]
                records = master_df[master_df["order_id"] == ord_id]
                tot_val = records["price"].sum() if "price" in records.columns else records["payment_value"].sum()
                status = records["order_status"].iloc[0] if "order_status" in records.columns else "N/A"
                date_val = records["order_purchase_timestamp"].iloc[0] if "order_purchase_timestamp" in records.columns else "N/A"

                summary = {
                    "Entity Type": "Order Details",
                    "Order ID": ord_id,
                    "Total Value": f"${tot_val:,.2f}",
                    "Status": str(status).title(),
                    "Purchase Timestamp": str(date_val),
                    "Items Count": len(records)
                }
                return {"has_match": True, "query": query, "match_type": "Order", "summary": summary, "records": records}
        return {"has_match": False}

    def _search_seller(self, query: str, master_df: pd.DataFrame) -> Dict[str, Any]:
        if "seller_id" in master_df.columns:
            matches = master_df[master_df["seller_id"].astype(str).str.contains(query, case=False, na=False)]
            if not matches.empty:
                seller_id = matches["seller_id"].iloc[0]
                records = master_df[master_df["seller_id"] == seller_id]
                tot_rev = records["price"].sum() if "price" in records.columns else 0.0
                tot_items = len(records)
                state = records["seller_state"].iloc[0] if "seller_state" in records.columns else "N/A"

                summary = {
                    "Entity Type": "Seller Profile",
                    "Seller ID": seller_id,
                    "State": state,
                    "Total Revenue": f"${tot_rev:,.2f}",
                    "Total Items Sold": f"{tot_items:,}"
                }
                return {"has_match": True, "query": query, "match_type": "Seller", "summary": summary, "records": records.head(50)}
        return {"has_match": False}

    def _search_product(self, query: str, master_df: pd.DataFrame) -> Dict[str, Any]:
        if "product_id" in master_df.columns:
            matches = master_df[master_df["product_id"].astype(str).str.contains(query, case=False, na=False)]
            if not matches.empty:
                product_id = matches["product_id"].iloc[0]
                records = master_df[master_df["product_id"] == product_id]
                tot_rev = records["price"].sum() if "price" in records.columns else 0.0
                units = len(records)
                cat = records["product_category_name_english"].iloc[0] if "product_category_name_english" in records.columns else "General"

                summary = {
                    "Entity Type": "Product Catalog SKU",
                    "Product ID": product_id,
                    "Category": str(cat).replace("_", " ").title(),
                    "Total Revenue": f"${tot_rev:,.2f}",
                    "Units Sold": f"{units:,}"
                }
                return {"has_match": True, "query": query, "match_type": "Product", "summary": summary, "records": records.head(50)}
        return {"has_match": False}

    def _search_category(self, query: str, master_df: pd.DataFrame) -> Dict[str, Any]:
        cat_col = "product_category_name_english" if "product_category_name_english" in master_df.columns else ("product_category_name" if "product_category_name" in master_df.columns else None)
        if cat_col:
            matches = master_df[master_df[cat_col].astype(str).str.contains(query, case=False, na=False)]
            if not matches.empty:
                cat_name = matches[cat_col].iloc[0]
                records = master_df[master_df[cat_col] == cat_name]
                tot_rev = records["price"].sum() if "price" in records.columns else 0.0
                orders = records["order_id"].nunique() if "order_id" in records.columns else len(records)

                summary = {
                    "Entity Type": "Product Category",
                    "Category Name": str(cat_name).replace("_", " ").title(),
                    "Total Category Revenue": f"${tot_rev:,.2f}",
                    "Total Orders": f"{orders:,}"
                }
                return {"has_match": True, "query": query, "match_type": "Category", "summary": summary, "records": records.head(50)}
        return {"has_match": False}
