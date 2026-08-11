"""
Customer Demographics Engine for ECIP Phase 12.
Analyzes Geographic Distribution by State and City, Top Customer Locations,
and Geographic Revenue contribution.
Generates structures for Maps, Bar Charts, and Treemaps.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("ECIP.DemographicsEngine")

# Coordinates for top Brazilian Cities / States
BRAZIL_STATE_COORDS = {
    "SP": {"name": "São Paulo", "lat": -23.55, "lon": -46.63},
    "RJ": {"name": "Rio de Janeiro", "lat": -22.90, "lon": -43.17},
    "MG": {"name": "Minas Gerais", "lat": -19.92, "lon": -43.93},
    "RS": {"name": "Rio Grande do Sul", "lat": -30.03, "lon": -51.23},
    "PR": {"name": "Paraná", "lat": -25.42, "lon": -49.27},
    "SC": {"name": "Santa Catarina", "lat": -27.59, "lon": -48.54},
    "BA": {"name": "Bahia", "lat": -12.97, "lon": -38.51},
    "DF": {"name": "Distrito Federal", "lat": -15.78, "lon": -47.92},
    "ES": {"name": "Espírito Santo", "lat": -20.31, "lon": -40.31},
    "GO": {"name": "Goiás", "lat": -16.68, "lon": -49.25},
    "PE": {"name": "Pernambuco", "lat": -8.05, "lon": -34.88},
    "CE": {"name": "Ceará", "lat": -3.71, "lon": -38.54},
    "PA": {"name": "Pará", "lat": -1.45, "lon": -48.50},
    "MT": {"name": "Mato Grosso", "lat": -15.60, "lon": -56.09},
    "MA": {"name": "Maranhão", "lat": -2.53, "lon": -44.30},
    "MS": {"name": "Mato Grosso do Sul", "lat": -20.44, "lon": -54.64},
    "PB": {"name": "Paraíba", "lat": -7.11, "lon": -34.86},
    "PI": {"name": "Piauí", "lat": -5.09, "lon": -42.80},
    "RN": {"name": "Rio Grande do Norte", "lat": -5.79, "lon": -35.20},
    "AL": {"name": "Alagoas", "lat": -9.66, "lon": -35.73},
    "SE": {"name": "Sergipe", "lat": -10.91, "lon": -37.07},
    "TO": {"name": "Tocantins", "lat": -10.18, "lon": -48.33},
    "RO": {"name": "Rondônia", "lat": -8.76, "lon": -63.90},
    "AM": {"name": "Amazonas", "lat": -3.10, "lon": -60.02},
    "AC": {"name": "Acre", "lat": -9.97, "lon": -67.80},
    "AP": {"name": "Amapá", "lat": 0.03, "lon": -51.06},
    "RR": {"name": "Roraima", "lat": 2.82, "lon": -60.67}
}

class CustomerDemographicsEngine:
    """Engine for demographic breakdown by state, city, and geospatial mapping."""

    def get_state_distribution(self, master_df: pd.DataFrame, feature_store_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Computes Customer Count, Total Revenue, and Lat/Lon coordinates per State."""
        if master_df.empty and (feature_store_df is None or feature_store_df.empty):
            return pd.DataFrame(columns=["State", "State_Name", "Customer_Count", "Total_Revenue", "Lat", "Lon"])

        df_target = master_df if not master_df.empty else feature_store_df
        state_col = "customer_state" if "customer_state" in df_target.columns else ("seller_state" if "seller_state" in df_target.columns else None)

        if not state_col:
            return pd.DataFrame(columns=["State", "State_Name", "Customer_Count", "Total_Revenue", "Lat", "Lon"])

        val_col = "price" if "price" in df_target.columns else ("total_spending" if "total_spending" in df_target.columns else state_col)
        cust_col = "customer_unique_id" if "customer_unique_id" in df_target.columns else "customer_id"

        agg = df_target.groupby(state_col).agg(
            Customer_Count=(cust_col, "nunique") if cust_col in df_target.columns else (state_col, "count"),
            Total_Revenue=(val_col, "sum") if val_col != state_col else (state_col, "count")
        ).reset_index()

        agg.rename(columns={state_col: "State"}, inplace=True)

        names, lats, lons = [], [], []
        for st in agg["State"]:
            st_u = str(st).upper()
            info = BRAZIL_STATE_COORDS.get(st_u, {"name": st_u, "lat": -14.23, "lon": -51.92})
            names.append(info["name"])
            lats.append(info["lat"])
            lons.append(info["lon"])

        agg["State_Name"] = names
        agg["Lat"] = lats
        agg["Lon"] = lons

        return agg.sort_values(by="Customer_Count", ascending=False)

    def get_city_distribution(self, master_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """Computes Top Cities by Customer Count and Total Sales."""
        if master_df.empty or "customer_city" not in master_df.columns:
            return pd.DataFrame(columns=["City", "Customer_Count", "Total_Revenue"])

        val_col = "price" if "price" in master_df.columns else "payment_value"
        cust_col = "customer_unique_id" if "customer_unique_id" in master_df.columns else "customer_id"

        agg = master_df.groupby("customer_city").agg(
            Customer_Count=(cust_col, "nunique") if cust_col in master_df.columns else ("customer_city", "count"),
            Total_Revenue=(val_col, "sum") if val_col in master_df.columns else ("customer_city", "count")
        ).reset_index()

        agg.rename(columns={"customer_city": "City"}, inplace=True)
        agg["City"] = agg["City"].astype(str).str.title()
        return agg.sort_values(by="Customer_Count", ascending=False).head(top_n)

    def get_geographic_revenue_treemap(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """Computes State -> City Geographic Revenue Treemap structure."""
        if master_df.empty:
            return pd.DataFrame(columns=["State", "City", "Total_Revenue", "Customer_Count"])

        state_col = "customer_state" if "customer_state" in master_df.columns else "seller_state"
        city_col = "customer_city" if "customer_city" in master_df.columns else state_col
        val_col = "price" if "price" in master_df.columns else "payment_value"

        if state_col not in master_df.columns or val_col not in master_df.columns:
            return pd.DataFrame(columns=["State", "City", "Total_Revenue", "Customer_Count"])

        agg = master_df.groupby([state_col, city_col]).agg(
            Total_Revenue=(val_col, "sum"),
            Customer_Count=("customer_unique_id", "nunique") if "customer_unique_id" in master_df.columns else (val_col, "count")
        ).reset_index()

        agg.rename(columns={state_col: "State", city_col: "City"}, inplace=True)
        agg["City"] = agg["City"].astype(str).str.title()
        return agg.sort_values(by="Total_Revenue", ascending=False)
