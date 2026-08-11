"""
Revenue Contribution & Pareto 80/20 Analysis Engine for ECIP Phase 12.
Computes Top 20 Customers by Revenue, Customer Revenue Distribution, Quantile Segments,
and Pareto Cumulative Revenue % vs Cumulative Customer % chart data.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("ECIP.RevenueContributionEngine")

class RevenueContributionEngine:
    """Engine for revenue contribution, customer ranking, and Pareto (80/20) analysis."""

    def get_top_customers_by_revenue(
        self,
        master_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """Computes Top N Customers sorted by total spending."""
        if master_df.empty and (feature_store_df is None or feature_store_df.empty):
            return pd.DataFrame(columns=["Customer_ID", "Total_Spending", "Total_Orders", "Avg_Order_Value", "State"])

        if not master_df.empty and "customer_unique_id" in master_df.columns:
            val_col = "price" if "price" in master_df.columns else "payment_value"
            state_col = "customer_state" if "customer_state" in master_df.columns else None

            agg = master_df.groupby("customer_unique_id").agg(
                Total_Spending=(val_col, "sum"),
                Total_Orders=("order_id", "nunique") if "order_id" in master_df.columns else (val_col, "count"),
                State=(state_col, "first") if state_col else ("customer_unique_id", lambda x: "N/A")
            ).reset_index()

            agg["Avg_Order_Value"] = agg["Total_Spending"] / agg["Total_Orders"].clip(lower=1)
            agg.rename(columns={"customer_unique_id": "Customer_ID"}, inplace=True)
            return agg.sort_values(by="Total_Spending", ascending=False).head(top_n)

        elif feature_store_df is not None and not feature_store_df.empty:
            df = feature_store_df.copy()
            id_col = "customer_unique_id" if "customer_unique_id" in df.columns else df.columns[0]
            spend_col = "total_spending" if "total_spending" in df.columns else ("historical_clv" if "historical_clv" in df.columns else df.columns[1])
            orders_col = "total_orders" if "total_orders" in df.columns else spend_col

            agg = df[[id_col, spend_col, orders_col]].copy()
            agg.columns = ["Customer_ID", "Total_Spending", "Total_Orders"]
            agg["Avg_Order_Value"] = agg["Total_Spending"] / agg["Total_Orders"].clip(lower=1)
            agg["State"] = "N/A"
            return agg.sort_values(by="Total_Spending", ascending=False).head(top_n)

        return pd.DataFrame(columns=["Customer_ID", "Total_Spending", "Total_Orders", "Avg_Order_Value", "State"])

    def get_pareto_analysis(
        self,
        master_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Computes Pareto (80/20) Curve data:
        - Customer_Percentile (0% to 100%)
        - Cumulative_Revenue_Percent (0% to 100%)
        - Summary stat (e.g. "Top 20% of customers generate 74.5% of total revenue").
        """
        if master_df.empty and (feature_store_df is None or feature_store_df.empty):
            return {
                "pareto_df": pd.DataFrame(columns=["Customer_Percentile", "Cumulative_Revenue_Pct"]),
                "summary_stat": "No data available for Pareto analysis.",
                "top_20_rev_pct": 0.0
            }

        # Calculate spending per customer
        if not master_df.empty and "customer_unique_id" in master_df.columns:
            val_col = "price" if "price" in master_df.columns else "payment_value"
            spending = master_df.groupby("customer_unique_id")[val_col].sum().sort_values(ascending=False).values
        elif feature_store_df is not None and not feature_store_df.empty and "total_spending" in feature_store_df.columns:
            spending = feature_store_df["total_spending"].dropna().sort_values(ascending=False).values
        else:
            return {
                "pareto_df": pd.DataFrame(columns=["Customer_Percentile", "Cumulative_Revenue_Pct"]),
                "summary_stat": "No data available for Pareto analysis.",
                "top_20_rev_pct": 0.0
            }

        total_rev = spending.sum()
        if total_rev == 0 or len(spending) == 0:
            return {
                "pareto_df": pd.DataFrame(columns=["Customer_Percentile", "Cumulative_Revenue_Pct"]),
                "summary_stat": "Total revenue is zero.",
                "top_20_rev_pct": 0.0
            }

        cum_rev = np.cumsum(spending)
        cum_rev_pct = (cum_rev / total_rev) * 100.0
        cust_pct = (np.arange(1, len(spending) + 1) / len(spending)) * 100.0

        # Sample 50 points evenly for high-performance chart rendering
        step = max(1, len(spending) // 50)
        idx_sample = list(range(0, len(spending), step))
        if (len(spending) - 1) not in idx_sample:
            idx_sample.append(len(spending) - 1)

        pareto_df = pd.DataFrame({
            "Customer_Percentile": cust_pct[idx_sample],
            "Cumulative_Revenue_Pct": cum_rev_pct[idx_sample]
        })

        # Calculate exact top 20% revenue share
        top_20_count = max(1, int(len(spending) * 0.20))
        top_20_rev = spending[:top_20_count].sum()
        top_20_rev_pct = round((top_20_rev / total_rev) * 100.0, 1)

        summary_stat = f"💡 **Pareto Principle (80/20)**: The top **20%** of customer accounts generate **{top_20_rev_pct}%** of gross enterprise revenue."

        return {
            "pareto_df": pareto_df,
            "summary_stat": summary_stat,
            "top_20_rev_pct": top_20_rev_pct
        }

    def get_revenue_quantile_segments(
        self,
        master_df: pd.DataFrame,
        feature_store_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Categorizes customers into Top 1%, Top 5%, Top 20%, Next 30%, and Bottom 50% revenue tiers."""
        if not master_df.empty and "customer_unique_id" in master_df.columns:
            val_col = "price" if "price" in master_df.columns else "payment_value"
            spending = master_df.groupby("customer_unique_id")[val_col].sum().sort_values(ascending=False)
        elif feature_store_df is not None and not feature_store_df.empty and "total_spending" in feature_store_df.columns:
            spending = feature_store_df["total_spending"].dropna().sort_values(ascending=False)
        else:
            return pd.DataFrame(columns=["Segment", "Customer_Count", "Total_Revenue", "Revenue_Share_Pct"])

        n = len(spending)
        if n == 0:
            return pd.DataFrame(columns=["Segment", "Customer_Count", "Total_Revenue", "Revenue_Share_Pct"])

        total_rev = spending.sum()
        p1 = max(1, int(n * 0.01))
        p5 = max(p1 + 1, int(n * 0.05))
        p20 = max(p5 + 1, int(n * 0.20))
        p50 = max(p20 + 1, int(n * 0.50))

        s1_rev = spending.iloc[:p1].sum()
        s2_rev = spending.iloc[p1:p5].sum()
        s3_rev = spending.iloc[p5:p20].sum()
        s4_rev = spending.iloc[p20:p50].sum()
        s5_rev = spending.iloc[p50:].sum()

        segments = [
            {"Segment": "Top 1% VIPs", "Customer_Count": p1, "Total_Revenue": s1_rev},
            {"Segment": "Top 5% High-Value", "Customer_Count": p5 - p1, "Total_Revenue": s2_rev},
            {"Segment": "Top 20% Core Buyers", "Customer_Count": p20 - p5, "Total_Revenue": s3_rev},
            {"Segment": "Next 30% Moderate Buyers", "Customer_Count": p50 - p20, "Total_Revenue": s4_rev},
            {"Segment": "Bottom 50% Long Tail", "Customer_Count": n - p50, "Total_Revenue": s5_rev}
        ]

        df_seg = pd.DataFrame(segments)
        df_seg["Revenue_Share_Pct"] = (df_seg["Total_Revenue"] / max(total_rev, 1.0)) * 100.0
        return df_seg
