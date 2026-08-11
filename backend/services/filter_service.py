"""
Multi-Dimensional Filter & Query Engine for ECIP Executive Dashboard.
Filters transaction and customer datasets dynamically by Date Range, State, Product Category,
Seller, Payment Method, and Customer Segment.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from backend.cache.dashboard_cache import dashboard_cache
from utils.logger import setup_logger

logger = setup_logger("ECIP.FilterService")

class FilterService:
    """Multi-criteria query engine for slicing and dicing enterprise datasets."""

    def filter_executive_data(
        self,
        master_df: pd.DataFrame,
        feature_store_df: pd.DataFrame = None,
        date_range: Optional[Tuple[Any, Any]] = None,
        states: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sellers: Optional[List[str]] = None,
        payment_methods: Optional[List[str]] = None,
        customer_segments: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies filter options to master_df and aligns feature_store_df accordingly.

        Returns:
            Tuple of (filtered_master_df, filtered_feature_store_df)
        """
        if master_df.empty:
            return pd.DataFrame(), pd.DataFrame() if feature_store_df is None else pd.DataFrame()

        fs_cols = sorted(list(feature_store_df.columns)) if feature_store_df is not None and not feature_store_df.empty else []
        params = {
            "date_range": date_range,
            "states": states,
            "categories": categories,
            "sellers": sellers,
            "payment_methods": payment_methods,
            "customer_segments": customer_segments,
            "fs_cols": fs_cols
        }

        # Check Cache
        cached = dashboard_cache.get("filtered_executive_data", params)
        if cached is not None:
            return cached

        filtered_master = master_df.copy()

        # 1. Date Range Filter
        if date_range and len(date_range) == 2 and date_range[0] is not None and date_range[1] is not None:
            start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            date_col = None
            for candidate in ["order_purchase_timestamp", "order_approved_at", "purchase_date"]:
                if candidate in filtered_master.columns:
                    date_col = candidate
                    break
            
            if date_col:
                filtered_master[date_col] = pd.to_datetime(filtered_master[date_col], errors="coerce")
                filtered_master = filtered_master[
                    (filtered_master[date_col] >= start_dt) &
                    (filtered_master[date_col] <= end_dt)
                ]

        # 2. State Filter
        if states and len(states) > 0:
            state_cols = [c for c in ["customer_state", "seller_state"] if c in filtered_master.columns]
            if state_cols:
                mask = filtered_master[state_cols[0]].isin(states)
                for sc in state_cols[1:]:
                    mask |= filtered_master[sc].isin(states)
                filtered_master = filtered_master[mask]

        # 3. Product Category Filter
        if categories and len(categories) > 0:
            cat_col = None
            for candidate in ["product_category_name_english", "product_category_name"]:
                if candidate in filtered_master.columns:
                    cat_col = candidate
                    break
            if cat_col:
                filtered_master = filtered_master[filtered_master[cat_col].isin(categories)]

        # 4. Seller Filter
        if sellers and len(sellers) > 0:
            if "seller_id" in filtered_master.columns:
                filtered_master = filtered_master[filtered_master["seller_id"].isin(sellers)]

        # 5. Payment Method Filter
        if payment_methods and len(payment_methods) > 0:
            pmt_col = None
            for candidate in ["payment_type", "preferred_payment_method"]:
                if candidate in filtered_master.columns:
                    pmt_col = candidate
                    break
            if pmt_col:
                filtered_master = filtered_master[filtered_master[pmt_col].isin(payment_methods)]

        # 6. Customer Segment Filter
        if customer_segments and len(customer_segments) > 0:
            seg_col = None
            for candidate in ["rfm_segment", "spending_tier", "clv_value_tier", "customer_segment"]:
                if candidate in filtered_master.columns:
                    seg_col = candidate
                    break
            if seg_col:
                filtered_master = filtered_master[filtered_master[seg_col].isin(customer_segments)]

        # Align Feature Store if provided
        filtered_fs = pd.DataFrame()
        if feature_store_df is not None and not feature_store_df.empty:
            filtered_fs = feature_store_df.copy()
            if "customer_unique_id" in filtered_master.columns and "customer_unique_id" in filtered_fs.columns:
                valid_cust_ids = set(filtered_master["customer_unique_id"].dropna().unique())
                filtered_fs = filtered_fs[filtered_fs["customer_unique_id"].isin(valid_cust_ids)]
            
            # Apply Customer Segment filter directly on FS if applicable
            if customer_segments and len(customer_segments) > 0:
                fs_seg_cols = [c for c in ["rfm_segment", "spending_tier", "clv_value_tier"] if c in filtered_fs.columns]
                if fs_seg_cols:
                    mask = filtered_fs[fs_seg_cols[0]].isin(customer_segments)
                    for sc in fs_seg_cols[1:]:
                        mask |= filtered_fs[sc].isin(customer_segments)
                    filtered_fs = filtered_fs[mask]

        result = (filtered_master, filtered_fs)
        dashboard_cache.set("filtered_executive_data", result, params, ttl=1800)
        logger.debug(f"Filtered Executive Master data shape: {filtered_master.shape}, FS shape: {filtered_fs.shape}")
        return result

    def extract_filter_options(self, master_df: pd.DataFrame, feature_store_df: pd.DataFrame = None) -> Dict[str, List[str]]:
        """Extracts unique dropdown choices for state, category, payment method, customer segment, seller list."""
        if master_df.empty:
            return {
                "states": [],
                "categories": [],
                "payment_methods": [],
                "customer_segments": [],
                "sellers": []
            }

        # States
        states = set()
        for col in ["customer_state", "seller_state"]:
            if col in master_df.columns:
                states.update(master_df[col].dropna().unique())

        # Categories
        categories = set()
        for col in ["product_category_name_english", "product_category_name"]:
            if col in master_df.columns:
                categories.update(master_df[col].dropna().unique())

        # Payment Methods
        pmts = set()
        for col in ["payment_type", "preferred_payment_method"]:
            if col in master_df.columns:
                pmts.update(master_df[col].dropna().unique())

        # Customer Segments
        segs = set()
        for col in ["rfm_segment", "spending_tier", "clv_value_tier"]:
            if col in master_df.columns:
                segs.update(master_df[col].dropna().unique())
            if feature_store_df is not None and not feature_store_df.empty and col in feature_store_df.columns:
                segs.update(feature_store_df[col].dropna().unique())

        # Sellers
        sellers = set()
        if "seller_id" in master_df.columns:
            sellers.update(master_df["seller_id"].dropna().unique())

        return {
            "states": sorted(list(states)),
            "categories": sorted(list(categories)),
            "payment_methods": sorted(list(pmts)),
            "customer_segments": sorted(list(segs)),
            "sellers": sorted(list(sellers))
        }
