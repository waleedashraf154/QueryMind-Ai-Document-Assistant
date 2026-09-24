import importlib.util
from pathlib import Path

QUERY_PATH = Path(__file__).parents[1] / "rag_pipeline" / "query.py"
STRUCT_PATH = Path(__file__).parents[1] / "rag_pipeline" / "document_structure.py"

# Import package normally when available.
from backend.rag_pipeline.query import _is_reference_count_query, _is_reference_list_query
from backend.rag_pipeline.document_structure import extract_reference_entries


def test_reference_count_variants():
    positives = [
        "How many references are in the paper?",
        "Tell me how many references",
        "Check again and tell me how many references",
        "What is the total number of references in this document?",
        "How many bibliography entries are there?",
    ]
    assert all(_is_reference_count_query(q) for q in positives)
    negatives = [
        "Explain the reference section",
        "Tell me reference 13",
        "What does the paper say about references?",
    ]
    assert not any(_is_reference_count_query(q) for q in negatives)


def test_reference_list_variants():
    positives = [
        "List all references",
        "Show me the bibliography",
        "Give me the complete reference list",
    ]
    assert all(_is_reference_list_query(q) for q in positives)


def test_reference_extraction_from_generic_text():
    text = """\nIntroduction\nThis cites [1,2] and [38].\n\nReferences\n[1] Alpha Author. Paper A. 2020.\n[2] Beta Author. Paper B. 2021.\n[3] Gamma Author. Paper C. 2022.\n"""
    refs = extract_reference_entries(text)
    assert len(refs) == 3
    assert refs[0].startswith("[1]")
    assert refs[-1].startswith("[3]")
