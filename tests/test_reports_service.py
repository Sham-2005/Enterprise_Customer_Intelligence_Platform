"""
Unit & Integration Test Suite for Phase 19 - Enterprise Reports & Export Center.
Tests ReportsService catalog definition, PDF generation, Excel workbook export, CSV export,
report preview payloads, report history logging, and scheduled report configurations.
"""

import pytest
import pandas as pd
import numpy as np

from backend.services.reports_service import ReportsService

@pytest.fixture
def dummy_report_df():
    """Generates synthetic dataset for report generation."""
    data = {
        "order_id": [f"ORD_{i:04d}" for i in range(10)],
        "customer_unique_id": [f"CUST_{i:03d}" for i in range(10)],
        "price": [100.0 + i*10 for i in range(10)],
        "product_category_name_english": ["health_beauty"] * 10
    }
    return pd.DataFrame(data)


def test_report_catalog_structure():
    """Verifies catalog definition contains 15 reports across 4 categories."""
    service = ReportsService()
    catalog = service.get_report_catalog()

    assert len(catalog) >= 15
    categories = set(r["category"] for r in catalog)
    assert "Executive" in categories
    assert "Customer" in categories
    assert "AI" in categories
    assert "Technical" in categories


def test_report_preview_generation():
    """Verifies live report preview payload structure."""
    service = ReportsService()
    preview = service.generate_report_preview("rep_exec_summary")

    assert preview["report_id"] == "rep_exec_summary"
    assert "kpi_summary" in preview
    assert "executive_summary_text" in preview
    assert "sample_table" in preview


def test_report_pdf_generation():
    """Verifies PDF report file generation."""
    service = ReportsService()
    res = service.generate_and_save_report("rep_exec_summary", export_format="PDF")

    assert res["success"] is True
    assert res["filename"].endswith(".pdf")
    assert isinstance(res["file_bytes"], bytes)
    assert len(res["file_bytes"]) > 0


def test_report_excel_generation():
    """Verifies Excel workbook generation."""
    service = ReportsService()
    res = service.generate_and_save_report("rep_cust_analytics", export_format="EXCEL")

    assert res["success"] is True
    assert res["filename"].endswith(".excel") or res["filename"].endswith(".xlsx")
    assert isinstance(res["file_bytes"], bytes)


def test_report_csv_generation():
    """Verifies CSV report export."""
    service = ReportsService()
    res = service.generate_and_save_report("rep_ai_churn", export_format="CSV")

    assert res["success"] is True
    assert res["filename"].endswith(".csv")
    assert isinstance(res["file_bytes"], bytes)


def test_report_history_logging():
    """Verifies appending and reading report history JSON log."""
    service = ReportsService()
    res = service.generate_and_save_report("rep_exec_summary", export_format="PDF")
    
    history = service.get_report_history()
    assert len(history) > 0
    assert history[0]["report_id"] == "rep_exec_summary"
    assert history[0]["generated_by"] == "System"


def test_scheduled_report_config():
    """Verifies scheduled report configuration architecture."""
    service = ReportsService()
    cfg = service.get_scheduled_report_config()

    assert cfg["scheduler_enabled"] is False
    assert "available_schedules" in cfg
