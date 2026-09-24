from __future__ import annotations

import fitz

from backend.parsers.document_parser import extract_text
from backend.rag_pipeline.document_structure import (
    extract_numbered_headings,
    extract_numbered_headings_from_pdf_file,
    extract_numbered_section,
    split_numbered_sections,
)
from backend.rag_pipeline.query import _extract_structured_references, _parse_section_target, classify_query_intent
from backend.rag_pipeline.retrieval import _retrieve_named_section


PDF = "/mnt/data/complex data.pdf"


def test_real_pdf_has_split_nested_headings():
    text = extract_text(PDF)
    heads = extract_numbered_headings_from_pdf_file(PDF)
    nums = [h.number for h in heads]
    assert "3.2" in nums
    assert "3.2.1" in nums
    assert "3.2.2" in nums
    assert "3.2.3" in nums


def test_real_pdf_source_text_section_extraction_handles_split_heading():
    text = extract_text(PDF)
    content, heading = extract_numbered_section(text, requested_number="3.2.1")
    assert heading == "3.2.1. Scaled Dot-Product Attention"
    assert content is not None
    assert "Scaled Dot-Product Attention" in content
    assert "dot products" in content
    assert "Multi-Head Attention" not in content.split("3.2.2", 1)[0]


def test_real_pdf_parent_section_includes_descendants_and_stops_at_sibling():
    text = extract_text(PDF)
    content, heading = extract_numbered_section(text, requested_number="3.2")
    assert heading == "3.2. Attention"
    assert content is not None
    assert "3.2.1. Scaled Dot-Product Attention" in content
    assert "3.2.2. Multi-Head Attention" in content
    assert "3.2.3. Applications of Attention in our Model" in content
    assert "3.3. Position-wise Feed-Forward Networks" not in content


def test_real_pdf_indexing_stores_hierarchical_section_metadata():
    text = extract_text(PDF)
    chunks, metas = split_numbered_sections(text)
    nums = [m.get("section_number") for m in metas]
    assert "3.2" in nums
    assert "3.2.1" in nums
    assert "3.2.2" in nums
    assert "3.2.3" in nums


class FakeCollection:
    def __init__(self, docs, metas):
        self.docs, self.metas = docs, metas
        self.ids = [f"paper_{i}" for i in range(len(docs))]

    def get(self, include=None):
        return {"documents": self.docs, "metadatas": self.metas, "ids": self.ids}


def test_parent_named_retrieval_includes_nested_sections():
    text = extract_text(PDF)
    chunks, metas = split_numbered_sections(text)
    docs = [c for c, m in zip(chunks, metas) if m.get("section_number", "").startswith("3.2")]
    metas2 = [dict(m, source="complex data.pdf") for m in metas if m.get("section_number", "").startswith("3.2")]
    col = FakeCollection(docs, metas2)
    got, _, got_metas = _retrieve_named_section(col, target_source="complex data.pdf", section_number="3.2")
    got_nums = {m.get("section_number") for m in got_metas}
    assert "3.2" in got_nums
    assert "3.2.1" in got_nums
    assert "3.2.2" in got_nums
    assert "3.2.3" in got_nums
    assert len(got) >= 4


def test_query_variants_route_generically():
    variants = [
        "Tell me point no 3.2.1",
        "What is point no 3.2.2?",
        "Explain subsection 3.2.3",
        "Tell me heading 3.2.2",
    ]
    for q in variants:
        refs = _extract_structured_references(q)
        assert len(refs) == 1
        parsed = _parse_section_target(q)
        assert parsed and parsed.get("number")
    assert classify_query_intent("Tell me point no 3.2.2", [], ["complex data.pdf"]) == "section_extraction"
    assert classify_query_intent("Tell me heading 3.2.2", [], ["complex data.pdf"]) == "heading_lookup"
