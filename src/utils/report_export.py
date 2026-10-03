"""
Report Generation and Export Utility
Exports predictions, SWOT analyses, and strategic roadmaps to PDF, Markdown, and CSV.
"""

from typing import Dict, List, Optional, Any
import io
import os
import pandas as pd


def _format_probability_line(probability: Optional[float]) -> str:
    """
    Formats the probability line for a report. `probability` is legitimately
    None for regression tasks (there is no "success probability", only a
    predicted numeric score) — this must never be treated as a bug to crash
    on, just a different display case.
    """
    if probability is None:
        return "**Tahmini Skor:** Bu, bir olasılık (yüzde) değeri üretmeyen bir regresyon tahminidir — yukarıdaki tahmin değeri doğrudan modelin öngördüğü sayısal sonuçtur."
    prob_pct = probability * 100 if probability <= 1.0 else probability
    return f"**Tahmini Başarı Oranı:** **%{prob_pct:.1f}**"


def generate_markdown_report(
    product_dict: Dict[str, Any],
    prediction_label: str,
    probability: Optional[float],
    best_model_name: str,
    model_metric: str,
    analysis_markdown: str,
    source: str
) -> str:
    """Generates a complete, beautifully structured Markdown report."""
    probability_line = _format_probability_line(probability)

    features_table = "| Özellik | Değer |\n|---|---|\n" + "\n".join([
        f"| **{k}** | `{v}` |" for k, v in product_dict.items()
    ])

    report = f"""# 🚀 AI Product Success Predictor & Strategic Intelligence Report

**Üretim Kaynağı:** {source}
**Eğitilen En İyi Model:** {best_model_name} (Metrik Skoru: {model_metric})

---

## 🎯 1. Tahmin Özeti
- **Ürün Başarı Durumu:** `{prediction_label}`
- {probability_line}

---

## 📋 2. Analiz Edilen Ürün Özellikleri
{features_table}

---

## 🤖 3. Yapay Zeka Strateji & SWOT Raporu
{analysis_markdown}

---
*Rapor, AI Product Success Predictor AutoML ve LLM Danışman Motoru tarafından otomatik üretilmiştir.*
"""
    return report


def generate_pdf_report(
    product_dict: Dict[str, Any],
    prediction_label: str,
    probability: Optional[float],
    best_model_name: str,
    model_metric: str,
    analysis_text: str
) -> bytes:
    """Generates PDF binary stream of the analysis report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=12
        )
        heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#4338CA"),
            spaceBefore=14,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6
        )

        elements = []

        # Title
        elements.append(Paragraph("AI Product Success Predictor — Rapor", title_style))
        elements.append(Paragraph(f"<b>Model:</b> {best_model_name} | <b>Metrik:</b> {model_metric}", body_style))
        elements.append(Spacer(1, 10))

        # Prediction Summary Box
        if probability is None:
            prob_display = "N/A (regresyon)"
        else:
            prob_pct = probability * 100 if probability <= 1.0 else probability
            prob_display = f"%{prob_pct:.1f}"
        summary_data = [
            ["Tahmin Edilen Sonuç", "Başarı Olasılığı"],
            [str(prediction_label), prob_display]
        ]
        t = Table(summary_data, colWidths=[240, 240])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#EEF2FF")),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor("#1E1B4B")),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 12),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 14))

        # Features Table
        elements.append(Paragraph("Ürün Özellikleri", heading_style))
        table_rows = [["Özellik", "Değer"]]
        for k, v in list(product_dict.items())[:12]:
            table_rows.append([str(k), str(v)])
        
        feat_table = Table(table_rows, colWidths=[200, 280])
        feat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(feat_table)
        elements.append(Spacer(1, 14))

        # AI Insights Section
        elements.append(Paragraph("Yapay Zeka Stratejik Analiz ve Öneriler", heading_style))
        cleaned_text = analysis_text.replace("#", "").replace("*", "").replace("`", "")
        for line in cleaned_text.split("\n"):
            line = line.strip()
            if line:
                elements.append(Paragraph(line, body_style))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        print(f"PDF creation error: {e}")
        # Return fallback text stream
        return analysis_text.encode("utf-8")
