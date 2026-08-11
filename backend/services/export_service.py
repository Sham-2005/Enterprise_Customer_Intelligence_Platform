"""
Report Export Pipeline Service for ECIP Executive Dashboard.
Generates downloadable files in CSV, Excel (.xlsx), and PDF formats
from filtered executive datasets.
"""

import io
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("ECIP.ExportService")

class ExportService:
    """Multi-format data export generator for CSV, Excel, and PDF reports."""

    def export_to_csv(self, df: pd.DataFrame) -> bytes:
        """Exports dataframe to UTF-8 CSV bytes buffer."""
        if df.empty:
            return b"No data available for export"
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")

    def export_to_excel(self, df: pd.DataFrame, sheet_name: str = "Executive Summary") -> bytes:
        """Exports dataframe to Excel (.xlsx) binary buffer."""
        if df.empty:
            return b"No data available for export"

        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception as e:
            logger.warning(f"openpyxl failed or missing, falling back to basic xlsxwriter: {e}")
            try:
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as ex:
                logger.error(f"Excel export error: {ex}")
                # Fallback to CSV buffer if Excel engines unavailable
                return self.export_to_csv(df)

        return buffer.getvalue()

    def export_to_pdf(
        self,
        df: pd.DataFrame,
        report_title: str = "ECIP Executive Dashboard Report",
        kpi_metrics: Optional[Dict[str, Any]] = None,
        summary_text: Optional[str] = None
    ) -> bytes:
        """
        Generates formatted PDF report using ReportLab or HTML-to-PDF string buffer.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            story = []

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#0f172a"), spaceAfter=12)
            sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#475569"), spaceAfter=14)
            h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#1e293b"), spaceBefore=10, spaceAfter=8)
            body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor("#334155"), leading=12)

            # Title
            story.append(Paragraph(f"<b>{report_title}</b>", title_style))
            story.append(Paragraph(f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | ECIP BI Backend", sub_style))

            # Executive Summary Section
            if summary_text:
                story.append(Paragraph("<b>Executive Summary</b>", h2_style))
                story.append(Paragraph(summary_text, body_style))
                story.append(Spacer(1, 10))

            # KPI Summary Table
            if kpi_metrics:
                story.append(Paragraph("<b>Key Performance Indicators (KPIs)</b>", h2_style))
                kpi_table_data = [["Metric", "Value", "Trend Change", "Previous Period"]]
                for k, v in kpi_metrics.items():
                    kpi_table_data.append([
                        v.get("title", k),
                        str(v.get("value", "N/A")),
                        str(v.get("change_pct", "0.0%")),
                        str(v.get("previous_period_value", "N/A"))
                    ])

                t_kpi = Table(kpi_table_data, colWidths=[150, 120, 100, 140])
                t_kpi.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                ]))
                story.append(t_kpi)
                story.append(Spacer(1, 14))

            # Data Table (First 30 rows)
            story.append(Paragraph("<b>Filtered Transactions Data Sample (Top 30 Records)</b>", h2_style))
            if not df.empty:
                disp_df = df.head(30).copy()
                cols = disp_df.columns[:6]  # Display first 6 columns for width fit
                table_data = [[str(c).replace("_", " ").title() for c in cols]]

                for _, row in disp_df.iterrows():
                    table_data.append([str(row[c])[:20] for c in cols])

                t_data = Table(table_data, colWidths=[85] * len(cols))
                t_data.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                ]))
                story.append(t_data)
            else:
                story.append(Paragraph("No records found in current filter view.", body_style))

            doc.build(story)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"ReportLab PDF generation error: {e}", exc_info=True)
            # Fallback simple text PDF buffer format
            text_rep = f"PDF REPORT - {report_title}\n\nSummary:\n{summary_text}\n\nExported {len(df)} rows."
            return text_rep.encode("utf-8")
