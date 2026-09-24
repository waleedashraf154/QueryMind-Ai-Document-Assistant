from __future__ import annotations

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from backend.rag_pipeline.document_structure import extract_numbered_headings_from_pdf_file


def test_pdf_layout_extracts_nested_headings_and_rejects_numeric_body(tmp_path):
    pdf = tmp_path / "paper.pdf"
    c = canvas.Canvas(str(pdf), pagesize=letter)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 720, "3.2.1 Scaled Dot-Product Attention")
    c.setFont("Helvetica", 10)
    c.drawString(72, 700, "We compute the dot products of queries and keys.")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 660, "3.2.2 Multi-Head Attention")
    c.setFont("Helvetica", 10)
    c.drawString(72, 640, "We use multiple attention heads in parallel.")
    c.drawString(72, 610, "2014. English-French dataset consisting of 36M sentences")
    c.showPage()
    c.save()

    headings = [r.full_text for r in extract_numbered_headings_from_pdf_file(str(pdf))]
    assert "3.2.1. Scaled Dot-Product Attention" in headings
    assert "3.2.2. Multi-Head Attention" in headings
    assert not any(h.startswith("2014.") for h in headings)
