from __future__ import annotations

import backend.rag_pipeline.query as q
from backend.rag_pipeline.document_structure import extract_headings_from_source_text, extract_structural_headings_from_chunks


def test_specific_hierarchical_references_are_not_global_heading_queries():
    cases = [
        ("Tell me Point no 3.2.2", "section_extraction"),
        ("What is point no 3.2.2?", "section_extraction"),
        ("What is section 3.2.2?", "section_extraction"),
        ("Tell me Heading of 3.2.2", "heading_lookup"),
        ("What is the heading/title of 3.2.2?", "heading_lookup"),
        ("Tell me 3.2.2 heading", "heading_lookup"),
    ]
    for query, expected_mode in cases:
        assert not q._is_heading_extraction_query(query), query
        assert q.classify_query_intent(query, [], ["paper.pdf"]) == expected_mode


def test_global_heading_queries_stay_global():
    for query in (
        "Tell me all headings",
        "Give me the main headings",
        "List all numbered headings in the document",
        "What are the document headings?",
    ):
        assert q._is_heading_extraction_query(query)
        assert q.classify_query_intent(query, [], ["paper.pdf"]) == "heading_extraction"


def test_numbered_heading_extraction_filters_numeric_prose_and_table_payloads():
    text = """
1. Introduction
Introductory text.
2. Background
Background text.
3. Model Architecture
Architecture text.
3.2.1 Scaled Dot-Product Attention
3.2.2 Multi-Head Attention
2014. English-French dataset consisting of 36M sentences and split tokens into a 32000 word piece
1.0. · 1020
2.3. · 1019
4. Results
Results text.
"""
    assert extract_headings_from_source_text(text) == [
        "1. Introduction",
        "2. Background",
        "3. Model Architecture",
        "4. Results",
    ]


def test_chunk_fallback_filters_false_numeric_headings():
    chunks = [
        "# 1. Introduction\nIntro",
        "## 2. Background\nBackground",
        "3. Model Architecture\nArchitecture",
        "2014. English-French dataset consisting of 36M sentences and split tokens into a 32000 word piece",
        "1.0. · 1020",
        "4. Results\nResults",
    ]
    assert extract_structural_headings_from_chunks(chunks) == [
        "1. Introduction",
        "2. Background",
        "3. Model Architecture",
        "4. Results",
    ]
