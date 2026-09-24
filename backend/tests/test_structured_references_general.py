from __future__ import annotations

from backend.rag_pipeline.query import _extract_structured_references, _parse_section_target
from backend.rag_pipeline.document_structure import extract_numbered_headings
from backend.rag_pipeline.retrieval import _retrieve_structured_sections


class FakeCollection:
    def __init__(self, docs, metas, ids):
        self.docs, self.metas, self.ids = docs, metas, ids

    def get(self, include=None):
        return {"documents": self.docs, "metadatas": self.metas, "ids": self.ids}


def test_hierarchical_point_reference_parses_without_question_hardcoding():
    q = "Tell me Point no 3.2.1"
    assert _extract_structured_references(q) == [("point", "3.2.1")]
    assert _parse_section_target(q)["number"] == "3.2.1"


def test_hierarchical_point_reference_retrieves_exact_subsection():
    c = FakeCollection(
        [
            "3.2.1 Scaled Dot-Product Attention\nExact target content.",
            "3.2.2 Multi-Head Attention\nSibling content.",
            "3.3 Applications\nLater content.",
        ],
        [
            {"source": "paper.pdf", "section_number": "3.2.1", "section_title": "Scaled Dot-Product Attention"},
            {"source": "paper.pdf", "section_number": "3.2.2", "section_title": "Multi-Head Attention"},
            {"source": "paper.pdf", "section_number": "3.3", "section_title": "Applications"},
        ],
        ["paper_0", "paper_1", "paper_2"],
    )
    docs, _, _ = _retrieve_structured_sections(c, [("point", "3.2.1")], target_source="paper.pdf")
    assert len(docs) == 1
    assert docs[0].startswith("3.2.1 Scaled Dot-Product Attention")


def test_parent_section_can_use_descendant_metadata():
    c = FakeCollection(
        [
            "3.2.1 A\ncontent A",
            "3.2.2 B\ncontent B",
            "3.3 C\ncontent C",
        ],
        [
            {"source": "paper.pdf", "section_number": "3.2.1", "section_title": "A"},
            {"source": "paper.pdf", "section_number": "3.2.2", "section_title": "B"},
            {"source": "paper.pdf", "section_number": "3.3", "section_title": "C"},
        ],
        ["paper_0", "paper_1", "paper_2"],
    )
    docs, _, _ = _retrieve_structured_sections(c, [("section", "3.2")], target_source="paper.pdf")
    assert len(docs) == 2
    assert all(d.startswith("3.2.") for d in docs)
