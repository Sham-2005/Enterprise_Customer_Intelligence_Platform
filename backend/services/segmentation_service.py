"""
Unified Customer Segmentation & RFM Intelligence Service for ECIP Phase 13.
Coordinates Data Ingestion, Multi-Criteria Filtering, Segmentation KPI Engine,
Cluster Explorer, PCA 2D/3D Analytics, RFM Dashboard Engine, Persona Manager,
Marketing Intelligence Engine, Search Engine, Export Generator, and Caching.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from backend.services.data_service import DataService
from backend.services.filter_service import FilterService
from backend.services.export_service import ExportService
from backend.segmentation.segmentation_kpi_engine import SegmentationKPIEngine
from backend.segmentation.cluster_explorer_engine import ClusterExplorerEngine
from backend.segmentation.rfm_dashboard_engine import RFMDashboardEngine
from backend.segmentation.persona_manager import PersonaManager
from backend.segmentation.marketing_intelligence import MarketingIntelligenceEngine
from backend.cache.dashboard_cache import dashboard_cache
from utils.logger import setup_logger

logger = setup_logger("ECIP.SegmentationService")

class SegmentationService:
    """Enterprise service orchestrator for Customer Segmentation & RFM Intelligence Module."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.data_service = DataService(config_path)
        self.filter_service = FilterService()
        self.kpi_engine = SegmentationKPIEngine()
        self.cluster_engine = ClusterExplorerEngine()
        self.rfm_engine = RFMDashboardEngine()
        self.persona_manager = PersonaManager()
        self.marketing_engine = MarketingIntelligenceEngine()
        self.export_service = ExportService()

    def get_filter_options(self) -> Dict[str, Any]:
        """Fast cached extraction of filter options for Segmentation."""
        try:
            from dashboard.utils.cache_manager import extract_global_filter_options
            opts = extract_global_filter_options()
            opts["clusters"] = opts.get("customer_segments", [])
            opts["personas"] = ["Champions", "Loyal Customers", "VIP Power Buyers", "At Risk", "Hibernating"]
            opts["revenue_range"] = (0.0, 100000.0)
            opts["clv_range"] = (0.0, 100000.0)
            opts["churn_risk"] = ["High Risk", "Medium Risk", "Low Risk"]
            return opts
        except Exception:
            datasets = self.data_service.load_all_executive_datasets()
            return self.filter_service.extract_filter_options(
                datasets.get("master_dataset", pd.DataFrame()),
                datasets.get("feature_store", pd.DataFrame())
            )

    def get_segmentation_payload(
        self,
        date_range: Optional[Tuple[Any, Any]] = None,
        clusters: Optional[List[str]] = None,
        personas: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        revenue_range: Optional[Tuple[float, float]] = None,
        clv_range: Optional[Tuple[float, float]] = None,
        churn_risk: Optional[List[str]] = None,
        loyalty_score_min: Optional[float] = None,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Loads, filters, and compiles complete Customer Segmentation & RFM payload in a single cached execution.
        """
        start_time = time.time()
        logger.info("Executing Customer Segmentation & RFM Payload Request...")

        filter_params = {
            "date_range": date_range,
            "clusters": clusters,
            "personas": personas,
            "states": states,
            "categories": categories,
            "revenue_range": revenue_range,
            "clv_range": clv_range,
            "churn_risk": churn_risk,
            "loyalty_score_min": loyalty_score_min
        }
        logger.info(f"Segmentation Filter changes applied: {filter_params}")

        # Step 1: Load Datasets
        datasets = self.data_service.load_all_executive_datasets(force_reload=force_reload)
        master_df = datasets.get("master_dataset", pd.DataFrame())
        feature_store_df = datasets.get("feature_store", pd.DataFrame())
        rfm_df = datasets.get("rfm_dataset", pd.DataFrame())
        metrics_df = datasets.get("customer_metrics", pd.DataFrame())

        # Step 2: Apply Filters
        filtered_master, filtered_fs = self.filter_service.filter_executive_data(
            master_df=master_df,
            feature_store_df=feature_store_df,
            date_range=date_range,
            states=states,
            categories=categories,
            customer_segments=clusters or personas
        )

        # Apply CLV Range Filter
        if clv_range and len(clv_range) == 2 and not filtered_fs.empty:
            clv_col = "historical_clv" if "historical_clv" in filtered_fs.columns else "predicted_clv"
            if clv_col in filtered_fs.columns:
                filtered_fs = filtered_fs[(filtered_fs[clv_col] >= clv_range[0]) & (filtered_fs[clv_col] <= clv_range[1])]

        # Step 3: Compute 8 Segmentation KPIs
        kpis = self.kpi_engine.compute_kpis(filtered_fs, rfm_df, filtered_master)

        # Step 4: Cluster Overview & Details
        cluster_overview = self.cluster_engine.get_cluster_overview(filtered_fs, filtered_master)

        # Step 5: PCA 2D/3D Visualization Data
        pca_data_df = self.cluster_engine.get_pca_visualization_data(filtered_fs, sample_size=500)

        # Step 6: Side-by-Side Cluster Comparison Matrix
        cluster_comparison_df = self.cluster_engine.get_cluster_comparison_matrix(filtered_fs)

        # Step 7: RFM Dashboard Data
        rfm_quintiles = self.rfm_engine.get_rfm_quintiles_distribution(filtered_fs, rfm_df)
        rfm_heatmap_matrix = self.rfm_engine.get_rfm_heatmap_matrix(filtered_fs, rfm_df)
        rfm_segments_df = self.rfm_engine.get_rfm_segment_distribution(filtered_fs, rfm_df)

        # Step 8: Business Personas
        personas_summary = self.persona_manager.get_personas_summary(filtered_fs)

        # Step 9: Marketing Intelligence Cards
        marketing_recommendations = self.marketing_engine.generate_recommendations(filtered_fs, filtered_master)

        # Extract Available Filter Options
        filter_opts = self.filter_service.extract_filter_options(master_df, feature_store_df)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Segmentation & RFM Payload compiled in {elapsed_ms}ms (Filtered FS Rows: {len(filtered_fs):,})")

        return {
            "performance_ms": elapsed_ms,
            "datasets_status": self.data_service.get_dataset_status(),
            "filtered_fs_rows": len(filtered_fs),
            "filtered_master_df": filtered_master,
            "filtered_fs_df": filtered_fs,
            "kpis": kpis,
            "cluster_overview": cluster_overview,
            "pca_data": pca_data_df,
            "cluster_comparison": cluster_comparison_df,
            "rfm_dashboard": {
                "quintiles": rfm_quintiles,
                "heatmap_matrix": rfm_heatmap_matrix,
                "segments_distribution": rfm_segments_df
            },
            "personas": personas_summary,
            "marketing_recommendations": marketing_recommendations,
            "filter_options": filter_opts
        }

    def get_selected_cluster_details(self, cluster_name: str) -> Dict[str, Any]:
        """Returns deep-dive stats for a single selected cluster."""
        datasets = self.data_service.load_all_executive_datasets()
        fs_df = datasets.get("feature_store", pd.DataFrame())
        m_df = datasets.get("master_dataset", pd.DataFrame())
        return self.cluster_engine.get_cluster_details(cluster_name, fs_df, m_df)

    def search_segmentation_profile(self, query: str) -> Dict[str, Any]:
        """
        Searches by Customer ID, Cluster Name, or Persona.
        Returns matching customer/cluster profile and recommendations.
        """
        query_str = str(query).strip()
        logger.info(f"Segmentation Search requested for query: '{query_str}'")

        if not query_str:
            return {"has_match": False, "query": query_str, "message": "Search query is empty."}

        datasets = self.data_service.load_all_executive_datasets()
        fs_df = datasets.get("feature_store", pd.DataFrame())
        m_df = datasets.get("master_dataset", pd.DataFrame())

        if fs_df.empty:
            return {"has_match": False, "query": query_str, "message": "No active dataset loaded for search."}

        # 1. Search Customer ID
        for c in ["customer_unique_id", "customer_id"]:
            if c in fs_df.columns:
                matches = fs_df[fs_df[c].astype(str).str.contains(query_str, case=False, na=False)]
                if not matches.empty:
                    row = matches.iloc[0]
                    cust_id = row[c]
                    tot_orders = row.get("total_orders", 1)
                    tot_spending = row.get("total_spending", 0.0)
                    clv = row.get("historical_clv", tot_spending)
                    rfm_seg = row.get("rfm_segment", row.get("cluster_name", "Champions"))
                    churn_risk = "Low Risk" if row.get("churn_label", 0) == 0 else "High Risk"

                    profile = {
                        "Customer ID": cust_id,
                        "Cluster / Segment": str(rfm_seg).replace("_", " ").title(),
                        "Total Orders": f"{tot_orders:,}",
                        "Total Spending": f"${tot_spending:,.2f}",
                        "Average Order Value": f"${tot_spending / max(tot_orders, 1):,.2f}",
                        "Historical CLV": f"${clv:,.2f}",
                        "Churn Risk": churn_risk,
                        "Recommended Strategy": "VIP retention & double point rewards" if tot_spending > 500 else "Re-engagement email campaign"
                    }
                    return {"has_match": True, "query": query_str, "match_type": "Customer ID", "profile": profile, "records": matches.head(30)}

        # 2. Search Cluster Name or Persona
        for c in ["cluster_name", "rfm_segment", "spending_tier"]:
            if c in fs_df.columns:
                matches = fs_df[fs_df[c].astype(str).str.contains(query_str, case=False, na=False)]
                if not matches.empty:
                    c_name = matches[c].iloc[0]
                    sub = fs_df[fs_df[c] == c_name]
                    tot_cust = len(sub)
                    tot_rev = sub["total_spending"].sum() if "total_spending" in sub.columns else 0.0

                    profile = {
                        "Segment / Persona Name": str(c_name).replace("_", " ").title(),
                        "Total Customer Count": f"{tot_cust:,}",
                        "Total Segment Revenue": f"${tot_rev:,.2f}",
                        "Average Customer Spending": f"${tot_rev / max(tot_cust, 1):,.2f}",
                        "Recommended Action": "Deploy targeted segment-specific campaign blueprint."
                    }
                    return {"has_match": True, "query": query_str, "match_type": "Cluster / Persona", "profile": profile, "records": sub.head(30)}

        return {"has_match": False, "query": query_str, "message": f"No customer, cluster, or persona matching '{query_str}' found."}

    def generate_export_file(
        self,
        format_type: str,
        filtered_fs_df: pd.DataFrame,
        kpis: Optional[Dict[str, Any]] = None,
        summary_text: Optional[str] = None
    ) -> Tuple[bytes, str, str]:
        """Generates downloadable binary/text buffer for CSV, Excel, or PDF report."""
        fmt = str(format_type).lower()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        logger.info(f"Segmentation Export event triggered for format: {fmt}")

        if fmt == "csv":
            content = self.export_service.export_to_csv(filtered_fs_df)
            filename = f"ecip_segmentation_{timestamp}.csv"
            mime = "text/csv"
        elif fmt in ["excel", "xlsx"]:
            content = self.export_service.export_to_excel(filtered_fs_df, sheet_name="Segmentation & RFM")
            filename = f"ecip_segmentation_{timestamp}.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif fmt == "pdf":
            content = self.export_service.export_to_pdf(
                df=filtered_fs_df,
                report_title="ECIP Customer Segmentation & RFM Intelligence Report",
                kpi_metrics=kpis,
                summary_text=summary_text or "Customer Segmentation & RFM Intelligence Report detailing cluster distributions, 2D/3D PCA projections, business personas, and marketing recommendations."
            )
            filename = f"ecip_segmentation_report_{timestamp}.pdf"
            mime = "application/pdf"
        else:
            content = self.export_service.export_to_csv(filtered_fs_df)
            filename = f"ecip_segmentation_{timestamp}.csv"
            mime = "text/csv"

        return content, filename, mime

# Global Singleton Instance
segmentation_service = SegmentationService()
