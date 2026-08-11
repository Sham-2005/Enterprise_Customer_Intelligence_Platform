"""
Customer Segmentation & RFM Intelligence Package for ECIP.
Provides specialized engines for Segmentation KPIs, Cluster Exploration,
RFM Dashboards, Business Personas Management, and Automated Marketing Intelligence.
"""

from backend.segmentation.segmentation_kpi_engine import SegmentationKPIEngine
from backend.segmentation.cluster_explorer_engine import ClusterExplorerEngine
from backend.segmentation.rfm_dashboard_engine import RFMDashboardEngine
from backend.segmentation.persona_manager import PersonaManager
from backend.segmentation.marketing_intelligence import MarketingIntelligenceEngine

__all__ = [
    "SegmentationKPIEngine",
    "ClusterExplorerEngine",
    "RFMDashboardEngine",
    "PersonaManager",
    "MarketingIntelligenceEngine"
]
