"""
Unified Executive Dashboard Backend Facade Service for ECIP.
Coordinates Data Ingestion, Filter Engine, KPI Engine, Analytics Engine,
Global Search Engine, Export Pipeline, Caching, and Logging.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from backend.services.data_service import DataService
from backend.services.filter_service import FilterService
from backend.services.kpi_service import KPIService
from backend.services.analytics_service import AnalyticsService
from backend.services.search_service import SearchService
from backend.services.export_service import ExportService
from backend.cache.dashboard_cache import dashboard_cache
from utils.logger import setup_logger

logger = setup_logger("ECIP.ExecutiveBackend")

class ExecutiveDashboardBackend:
    """Enterprise backend engine orchestrator for the Executive Dashboard."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.data_service = DataService(config_path)
        self.filter_service = FilterService()
        self.kpi_service = KPIService()
        self.analytics_service = AnalyticsService()
        self.search_service = SearchService()
        self.export_service = ExportService()

    def get_filter_options(self) -> Dict[str, Any]:
        """Fast cached extraction of filter options without computing entire payload."""
        try:
            from dashboard.utils.cache_manager import extract_global_filter_options
            return extract_global_filter_options()
        except Exception:
            all_datasets = self.data_service.load_all_executive_datasets()
            return self.filter_service.extract_filter_options(
                all_datasets.get("master_dataset", pd.DataFrame()),
                all_datasets.get("feature_store", pd.DataFrame())
            )

    def get_dashboard_payload(
        self,
        date_range: Optional[Tuple[Any, Any]] = None,
        states: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sellers: Optional[List[str]] = None,
        payment_methods: Optional[List[str]] = None,
        customer_segments: Optional[List[str]] = None,
        revenue_granularity: str = "Monthly",
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Loads, filters, and computes all executive dashboard components in a single cached execution flow.
        Tracks performance metrics and logs operations.
        """
        start_time = time.time()
        logger.info("Executing Executive Dashboard Backend Payload Request...")

        filter_params = {
            "date_range": date_range,
            "states": states,
            "categories": categories,
            "sellers": sellers,
            "payment_methods": payment_methods,
            "customer_segments": customer_segments,
            "revenue_granularity": revenue_granularity
        }
        logger.info(f"Filter changes applied: {filter_params}")

        # Step 1: Load Datasets
        all_datasets = self.data_service.load_all_executive_datasets(force_reload=force_reload)
        master_df = all_datasets.get("master_dataset", pd.DataFrame())
        feature_store_df = all_datasets.get("feature_store", pd.DataFrame())

        # Step 2: Apply Multi-Dimensional Filters
        filtered_master, filtered_fs = self.filter_service.filter_executive_data(
            master_df=master_df,
            feature_store_df=feature_store_df,
            date_range=date_range,
            states=states,
            categories=categories,
            sellers=sellers,
            payment_methods=payment_methods,
            customer_segments=customer_segments
        )

        # Step 3: Compute KPIs
        kpis = self.kpi_service.compute_all_kpis(filtered_master, filtered_fs)

        # Step 4: Compute Analytics Charts Data
        rev_trend_df = self.analytics_service.get_revenue_trend(filtered_master, granularity=revenue_granularity)
        growth_df = self.analytics_service.get_customer_growth(filtered_master, filtered_fs)
        category_treemap_df = self.analytics_service.get_revenue_by_category_treemap(filtered_master)
        state_map_df = self.analytics_service.get_revenue_by_state_map(filtered_master)
        pmt_dist_df = self.analytics_service.get_payment_method_distribution(filtered_master)
        order_status_df = self.analytics_service.get_order_status_distribution(filtered_master)
        top_products_df = self.analytics_service.get_top_selling_products(filtered_master, top_n=10)
        ratings_hist_df = self.analytics_service.get_customer_ratings_histogram(filtered_master)

        # Step 5: Insights & Summary
        recent_insights = self.analytics_service.generate_recent_business_insights(filtered_master, filtered_fs)
        executive_summary = self.analytics_service.generate_executive_summary_text(filtered_master, filtered_fs)

        # Filter Options for UI Dropdowns
        filter_options = self.filter_service.extract_filter_options(master_df, feature_store_df)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Dashboard Payload compiled successfully in {elapsed_ms}ms (Filtered Master Rows: {len(filtered_master):,})")

        return {
            "performance_ms": elapsed_ms,
            "datasets_status": self.data_service.get_dataset_status(),
            "filtered_master_rows": len(filtered_master),
            "filtered_master_df": filtered_master,
            "filtered_fs_df": filtered_fs,
            "kpis": kpis,
            "charts": {
                "revenue_trend": rev_trend_df,
                "customer_growth": growth_df,
                "category_treemap": category_treemap_df,
                "state_map": state_map_df,
                "payment_distribution": pmt_dist_df,
                "order_status": order_status_df,
                "top_products": top_products_df,
                "ratings_histogram": ratings_hist_df
            },
            "recent_insights": recent_insights,
            "executive_summary": executive_summary,
            "filter_options": filter_options
        }

    def execute_global_search(self, query: str) -> Dict[str, Any]:
        """Executes universal search across Customer, Order, Seller, and Product entities."""
        logger.info(f"Global search requested for query: '{query}'")
        t0 = time.time()
        datasets = self.data_service.load_all_executive_datasets()
        master_df = datasets.get("master_dataset", pd.DataFrame())
        fs_df = datasets.get("feature_store", pd.DataFrame())

        result = self.search_service.search(query, master_df, fs_df)
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        logger.info(f"Global search completed in {elapsed_ms}ms. Match Type: {result.get('match_type')}")
        return result

    def generate_export_file(
        self,
        format_type: str,
        filtered_master_df: pd.DataFrame,
        kpi_metrics: Optional[Dict[str, Any]] = None,
        summary_text: Optional[str] = None
    ) -> Tuple[bytes, str, str]:
        """
        Generates downloadable binary/text buffer for CSV, Excel, or PDF report.

        Returns:
            Tuple of (file_bytes, filename, mime_type)
        """
        fmt = str(format_type).lower()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        logger.info(f"Export event triggered for format: {fmt}")

        if fmt == "csv":
            content = self.export_service.export_to_csv(filtered_master_df)
            filename = f"ecip_executive_summary_{timestamp}.csv"
            mime = "text/csv"
        elif fmt in ["excel", "xlsx"]:
            content = self.export_service.export_to_excel(filtered_master_df)
            filename = f"ecip_executive_summary_{timestamp}.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif fmt == "pdf":
            content = self.export_service.export_to_pdf(
                df=filtered_master_df,
                report_title="ECIP Executive Intelligence Report",
                kpi_metrics=kpi_metrics,
                summary_text=summary_text
            )
            filename = f"ecip_executive_report_{timestamp}.pdf"
            mime = "application/pdf"
        else:
            content = self.export_service.export_to_csv(filtered_master_df)
            filename = f"ecip_executive_summary_{timestamp}.csv"
            mime = "text/csv"

        return content, filename, mime

# Global Singleton Instance
executive_backend = ExecutiveDashboardBackend()
