from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.parsers.document_parser import extract_text
from backend.rag_pipeline.chunking import chunk_text
from backend.rag_pipeline.document_structure import split_numbered_sections, extract_numbered_headings
from backend.rag_pipeline.query import (
    _parse_section_target,
    _extract_document_headings,
    _is_continuation_query,
    _resolve_continuation_question,
)
from backend.rag_pipeline.retrieval import _retrieve_named_section
from backend.rag_pipeline import service as service_mod

PDF = Path("/mnt/data/querymind large text file(1).pdf")


class FakeCollection:
    def __init__(self, docs, metas):
        self.docs = docs
        self.metas = metas

    def get(self, include=None):
        return {"documents": list(self.docs), "metadatas": list(self.metas), "ids": [f"file_{i}" for i in range(len(self.docs))]}

    def count(self):
        return len(self.docs)


def test_pdf_reading_order_keeps_first_two_headings_in_place():
    text = extract_text(str(PDF))
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    i1 = lines.index("1. Executive Overview")
    i2 = lines.index("2. Company Background")
    body_after_1 = "This document is a synthetic five-thousand-word test corpus designed for evaluating document question"
    assert i1 < lines.index(body_after_1) < i2


def test_numbered_document_has_exactly_15_headings():
    text = extract_text(str(PDF))
    headings = extract_numbered_headings(text)
    assert len(headings) == 15
    assert headings[0].full_text == "1. Executive Overview"
    assert headings[-1].full_text == "15. Conclusion and Test Notes"


def test_numbered_chunker_preserves_section_boundaries_and_numbers():
    text = extract_text(str(PDF))
    chunks, metas = split_numbered_sections(text)
    assert len(chunks) == len(metas)
    assert chunks
    assert all(not c.startswith("Section: ") for c in chunks)
    section9 = [m for m in metas if m.get("section_number") == "9"]
    assert len(section9) >= 2
    assert all(m.get("section_title") == "Customer Support Guide" for m in section9)


def test_chunk_text_prefers_numbered_report_path():
    text = extract_text(str(PDF))
    chunks = chunk_text(text)
    assert len(chunks) >= 15
    assert any(c.startswith("9. Customer Support Guide") for c in chunks)
    assert not any("Section: 9. Customer Support Guide" in c for c in chunks)


def test_section_query_strips_document_wrappers():
    parsed = _parse_section_target("Tell me customer support guide from the file only")
    assert parsed == {"type": "title", "number": None, "title": "customer support guide", "raw": "customer support guide"}
    parsed2 = _parse_section_target("Tell me about Data Processing and Storage from the file")
    assert parsed2["type"] == "title"
    assert parsed2["title"].casefold() == "data processing and storage"


def test_continuation_query_resolves_from_prior_section():
    q = "Tell me remaining more"
    assert _is_continuation_query(q)
    working, previous_user, previous_answer = _resolve_continuation_question(
        q,
        [
            {"role": "user", "content": "Tell me customer support guide from the file only"},
            {"role": "assistant", "content": "partial section answer"},
            {"role": "user", "content": q},
        ],
    )
    assert previous_user == "Tell me customer support guide from the file only"
    assert previous_answer == "partial section answer"
    assert "customer support guide" in working.casefold()
    assert "fresh" not in working.casefold() or "retrieve" in working.casefold()




def test_heading_query_phrasings_cover_numbered_variants():
    from backend.rag_pipeline.query import _is_heading_extraction_query

    assert _is_heading_extraction_query("Only Numbering Headings")
    assert _is_heading_extraction_query("Only Numbered Headings")
    assert _is_heading_extraction_query("Tell me all 15 headings")
    assert _is_heading_extraction_query("Give me all 15 numbered headings")
    assert _is_heading_extraction_query("Main Headings")


def test_numbered_headings_are_sorted_independent_of_chunk_order():
    docs = [
        "15. Conclusion and Test Notes\ncontent",
        "3. Atlas Product Definition\ncontent",
        "1. Executive Overview\ncontent",
        "10. Data Processing and Storage\ncontent",
        "2. Company Background\ncontent",
    ]
    assert _extract_document_headings(docs) == [
        "1. Executive Overview",
        "2. Company Background",
        "3. Atlas Product Definition",
        "10. Data Processing and Storage",
        "15. Conclusion and Test Notes",
    ]


def test_heading_extraction_is_deterministic_and_excludes_fake_section_labels():
    docs = [
        "1. Executive Overview\ncontent",
        "Section: PROFILE\nwrong",
        "2. Company Background\ncontent",
        "Section: REFERENCE\nwrong",
        "9. Customer Support Guide\ncontent",
        "Reference Table A Atlas configuration",
    ]
    heads = _extract_document_headings(docs)
    assert heads == ["1. Executive Overview", "2. Company Background", "9. Customer Support Guide"]


def test_named_section_retrieval_returns_all_chunks_in_order():
    docs = [
        "1. Executive Overview\na",
        "2. Company Background\nb",
        "2. Company Background\nc",
        "9. Customer Support Guide\nd",
        "9. Customer Support Guide\ne",
        "10. Data Processing and Storage\nf",
    ]
    metas = [
        {"source": "x.pdf", "section_number": "1", "section_title": "Executive Overview", "section_key": "executive overview", "chunk_index": 0},
        {"source": "x.pdf", "section_number": "2", "section_title": "Company Background", "section_key": "company background", "chunk_index": 1},
        {"source": "x.pdf", "section_number": "2", "section_title": "Company Background", "section_key": "company background", "chunk_index": 2},
        {"source": "x.pdf", "section_number": "9", "section_title": "Customer Support Guide", "section_key": "customer support guide", "chunk_index": 3},
        {"source": "x.pdf", "section_number": "9", "section_title": "Customer Support Guide", "section_key": "customer support guide", "chunk_index": 4},
        {"source": "x.pdf", "section_number": "10", "section_title": "Data Processing and Storage", "section_key": "data processing and storage", "chunk_index": 5},
    ]
    docs2, dists, metas2 = _retrieve_named_section(FakeCollection(docs, metas), target_source="x.pdf", section_number="9")
    assert docs2 == ["9. Customer Support Guide\nd", "9. Customer Support Guide\ne"]
    assert [m["chunk_index"] for m in metas2] == [3, 4]
    assert dists == [0.0, 0.0]


def test_service_section_mode_returns_complete_section_without_llm():
    text = extract_text(str(PDF))
    chunks, metas = split_numbered_sections(text)
    metas = [{**m, "source": "querymind large text file.pdf", "chunk_index": i} for i, m in enumerate(metas)]
    collection = FakeCollection(chunks, metas)
    fake_intent = SimpleNamespace(intent="document_query", confidence=1.0, evidence_type=None, topic=None, question_type=None)

    with patch.object(service_mod, "get_chroma_collection", return_value=collection), \
         patch.object(service_mod, "classify_intent", return_value=fake_intent):
        result = service_mod._answer_question_core(
            "test-chat",
            "Tell me customer support guide from the file only",
            selected_source="querymind large text file.pdf",
        )

    answer = result["answer"]
    assert answer.startswith("9. Customer Support Guide")
    assert "Reference Table A Atlas configuration" in answer
    assert "Reference Table D annual 2026 budget" in answer
    assert "10. Data Processing and Storage" not in answer


def test_service_continuation_uses_previous_section():
    text = extract_text(str(PDF))
    chunks, metas = split_numbered_sections(text)
    metas = [{**m, "source": "querymind large text file.pdf", "chunk_index": i} for i, m in enumerate(metas)]
    collection = FakeCollection(chunks, metas)
    fake_intent = SimpleNamespace(intent="document_query", confidence=1.0, evidence_type=None, topic=None, question_type=None)

    with patch.object(service_mod, "get_chroma_collection", return_value=collection), \
         patch.object(service_mod, "classify_intent", return_value=fake_intent):
        result = service_mod._answer_question_core(
            "test-chat",
            "Tell me remaining more",
            selected_source="querymind large text file.pdf",
            conversation_history=[
                {"role": "user", "content": "Tell me customer support guide from the file only"},
                {"role": "assistant", "content": "previous partial answer"},
                {"role": "user", "content": "Tell me remaining more"},
            ],
        )

    assert result["answer"].startswith("9. Customer Support Guide")
    assert "10. Data Processing and Storage" not in result["answer"]


def test_deep_relationship_question_reaches_normal_rag():
    collection = FakeCollection(
        ["3. Atlas Product Definition\nretrieval pipeline details", "6. Operations Metrics\nretrieval relevance and groundedness"],
        [
            {"source": "x.pdf", "section_number": "3", "section_title": "Atlas Product Definition", "chunk_index": 0},
            {"source": "x.pdf", "section_number": "6", "section_title": "Operations Metrics", "chunk_index": 1},
        ],
    )
    fake_intent = SimpleNamespace(intent="document_query", confidence=1.0, evidence_type="TEXT_EVIDENCE", topic=None, question_type="document_fact")
    fake_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded deep answer"))])

    with patch.object(service_mod, "get_chroma_collection", return_value=collection), \
         patch.object(service_mod, "classify_intent", return_value=fake_intent), \
         patch.object(service_mod, "_retrieve_relevant_chunks", return_value=(
             ["3. Atlas Product Definition\nretrieval pipeline details", "6. Operations Metrics\nretrieval relevance and groundedness"],
             [0.1, 0.2],
             collection.metas,
         )), \
         patch.object(service_mod, "_get_managed_prompt", return_value=("system", None)), \
         patch.object(service_mod, "_langfuse_track_generation", return_value=fake_response), \
         patch.object(service_mod, "_groq_client", object()):
        result = service_mod._answer_question_core(
            "deep-chat",
            "What is the relationship between the retrieval pipeline and the operational metrics?",
            selected_source="x.pdf",
        )

    assert result["answer"] == "Grounded deep answer"
    assert len(result["sources"]) >= 1




def test_heading_fallback_excludes_page_chrome_and_legacy_section_wrappers():
    docs = [
        "Section: PROFILE\nbody",
        "Page 5",
        "--- Page 6 ---",
        "Company Background",
        "Reference Table A – Atlas configuration",
        "End of test corpus.",
    ]
    assert _extract_document_headings(docs) == ["Company Background"]


def test_resume_inline_headings_are_recovered():
    from backend.rag_pipeline.chunking import _structure_aware_chunks

    text = (
        "MUHAMMAD WALEED INFO... ABOUT... "
        "PROFESSIONAL EXPERIENCE Customer Support Representative (Foodpanda) "
        "Mindbridge Private Limited NOV 2024 - MAY 2025 "
        "EDUCATION Matric Computer Science from PAK Angels Grammar School "
        "CONTACT +92 309 1442330"
    )
    chunks = _structure_aware_chunks(text)
    joined = "\n".join(chunks).lower()
    assert "section: professional experience" in joined
    assert "customer support representative" in joined
    assert "section: education" in joined
    assert "section: contact" in joined


def test_unique_natural_filename_alias():
    from backend.rag_pipeline.retrieval import _detect_target_source

    sources = ["Muhammad Waleed Resume.pdf", "querymind large text file.pdf", "other.pdf"]
    result = _detect_target_source("In resume.pdf file check", sources, document_sources=sources)
    assert result == "Muhammad Waleed Resume.pdf"


def test_general_heading_request_language_is_not_hardcoded():
    from backend.rag_pipeline.query import _is_heading_extraction_query, _parse_section_target
    variants = [
        "Tell me 15 headings",
        "list 15 headings",
        "what are the document headings",
        "show every numbered heading",
        "headings only please",
        "give me the main headings from this file",
    ]
    for question in variants:
        assert _is_heading_extraction_query(question), question
        assert _parse_section_target(question) is None, question


def test_numbered_source_heading_extraction_ignores_titles_and_page_chrome():
    from backend.rag_pipeline.document_structure import extract_headings_from_source_text
    source = """--- Page 1 ---\nQueryMind 5000-Word Test Document\n\n1. Executive Overview\nBody\n2. Company Background\nBody\nQueryMind synthetic test corpus\nPage 1\n--- Page 2 ---\n3. Atlas Product Definition\nBody\nEnd of test corpus.\n"""
    assert extract_headings_from_source_text(source) == [
        "1. Executive Overview",
        "2. Company Background",
        "3. Atlas Product Definition",
    ]


def test_chunk_heading_fallback_does_not_promote_prose_fragments():
    from backend.rag_pipeline.document_structure import extract_structural_headings_from_chunks
    docs = ["This\nMB\nquestions. The\n2. Real Heading\nBody"]
    assert extract_structural_headings_from_chunks(docs) == ["2. Real Heading"]



def test_specific_heading_number_is_not_global_heading_extraction():
    from backend.rag_pipeline.query import _is_heading_extraction_query, _is_specific_heading_lookup_query, _parse_section_target
    variants = [
        "Tell me heading 13",
        "Tell me heading no. 13",
        "Show me heading number 13",
        "What is heading 13?",
        "Give me the 13th heading",
    ]
    for q in variants:
        assert _is_specific_heading_lookup_query(q), q
        assert not _is_heading_extraction_query(q), q
        parsed = _parse_section_target(q)
        assert parsed and parsed.get("number") == "13", (q, parsed)


def test_point_number_maps_to_section_number():
    from backend.rag_pipeline.query import _parse_section_target, _is_heading_extraction_query
    q = "explain me point 13"
    assert not _is_heading_extraction_query(q)
    parsed = _parse_section_target(q)
    assert parsed and parsed.get("number") == "13", parsed



def test_specific_heading_explanation_is_section_query_not_global_outline():
    from backend.rag_pipeline.query import _is_heading_extraction_query, _parse_section_target
    for q in ["Explain heading 13", "Tell me about heading 13", "Explain me point 13"]:
        assert not _is_heading_extraction_query(q), q
        parsed = _parse_section_target(q)
        assert parsed and parsed.get("number") == "13", (q, parsed)


def _section_meta(source, number, title):
    return {"source": source, "section_number": str(number), "section_title": title, "type": "text"}


def test_generic_person_request_is_not_section_mode():
    from backend.rag_pipeline.query import classify_query_intent
    assert classify_query_intent("Give me Lukasz Kaiser", [], ["attention.pdf"]) == "document_qa"
    assert classify_query_intent("Give me details of Niki Parmar", [], ["attention.pdf"]) == "document_qa"
    assert classify_query_intent("Tell me about Lukasz Kaiser", [], ["attention.pdf"]) == "topic"


def test_explicit_section_wording_still_routes_to_section_mode():
    from backend.rag_pipeline.query import classify_query_intent
    assert classify_query_intent("Explain section 3", [], ["attention.pdf"]) == "section_extraction"
    assert classify_query_intent("Tell me heading 13", [], ["report.pdf"]) == "heading_lookup"


def test_generic_named_target_only_becomes_section_when_document_heading_matches(monkeypatch):
    from unittest.mock import MagicMock
    from backend.rag_pipeline.service import _answer_question_core
    from backend.rag_pipeline.intent_router import IntentClassification

    mock_collection = MagicMock()
    docs = [
        "1. Introduction\nIntro content",
        "2. Methods\nMethods content",
        "Lukasz Kaiser is an author listed in the paper.",
    ]
    metas = [
        _section_meta("attention.pdf", 1, "Introduction"),
        _section_meta("attention.pdf", 2, "Methods"),
        {"source": "attention.pdf", "type": "text", "section_number": "", "section_title": ""},
    ]
    mock_collection.count.return_value = 3
    mock_collection.get.return_value = {"documents": docs, "metadatas": metas, "ids": ["chunk_0", "chunk_1", "chunk_2"]}
    mock_collection.query.return_value = {
        "documents": [[docs[2]]],
        "metadatas": [[metas[2]]],
        "distances": [[0.1]],
        "ids": [["chunk_2"]],
    }

    monkeypatch.setattr("backend.rag_pipeline.service.get_chroma_collection", lambda _chat_id: mock_collection)
    monkeypatch.setattr("backend.rag_pipeline.service.classify_intent", lambda *args, **kwargs: IntentClassification(
        evidence_type="TEXT_EVIDENCE", intent="document_query", confidence=0.99,
        reason="test", topic=None, question_type="document_fact"
    ))

    def fake_rag(*args, **kwargs):
        return [docs[2]], [0.1], [metas[2]]

    monkeypatch.setattr("backend.rag_pipeline.service._retrieve_relevant_chunks", fake_rag)

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="Lukasz Kaiser is an author listed in the paper."))]
    fake_groq = MagicMock()
    fake_groq.chat.completions.create.return_value = fake_response
    monkeypatch.setattr("backend.rag_pipeline.generation._groq_client", fake_groq)
    monkeypatch.setattr("backend.rag_pipeline.service._groq_client", fake_groq)

    result = _answer_question_core("chat-test", "Give me Lukasz Kaiser")
    assert "Lukasz Kaiser" in result["answer"]


def test_generic_named_target_matches_real_section(monkeypatch):
    from unittest.mock import MagicMock
    from backend.rag_pipeline.service import _answer_question_core
    from backend.rag_pipeline.intent_router import IntentClassification

    mock_collection = MagicMock()
    docs = [
        "9. Customer Support Guide\nFirst support paragraph.",
        "9. Customer Support Guide\nSecond support paragraph.",
        "10. Data Processing and Storage\nOther section.",
    ]
    metas = [
        _section_meta("report.pdf", 9, "Customer Support Guide"),
        _section_meta("report.pdf", 9, "Customer Support Guide"),
        _section_meta("report.pdf", 10, "Data Processing and Storage"),
    ]
    mock_collection.count.return_value = 3
    mock_collection.get.return_value = {"documents": docs, "metadatas": metas, "ids": ["chunk_9_0", "chunk_9_1", "chunk_10_0"]}

    monkeypatch.setattr("backend.rag_pipeline.service.get_chroma_collection", lambda _chat_id: mock_collection)
    monkeypatch.setattr("backend.rag_pipeline.service.classify_intent", lambda *args, **kwargs: IntentClassification(
        evidence_type="TEXT_EVIDENCE", intent="document_query", confidence=0.99,
        reason="test", topic=None, question_type="document_fact"
    ))

    result = _answer_question_core("chat-test", "Tell me about Customer Support Guide")
    assert result["answer"].startswith("9. Customer Support Guide")
    assert "First support paragraph" in result["answer"]
    assert "Second support paragraph" in result["answer"]
    assert "Data Processing and Storage" not in result["answer"]


def test_screenshot_style_person_request_routes_to_document_qa():
    from backend.rag_pipeline.query import classify_query_intent
    assert classify_query_intent("Give me Lukasz Kaiser", [], ["attention_is_all_you_need.pdf"]) == "document_qa"
    assert classify_query_intent("Give me details of Niki Parmar", [], ["attention_is_all_you_need.pdf"]) == "document_qa"


def test_generic_named_target_does_not_trigger_section_mode_without_heading_match():
    from backend.rag_pipeline.service import _generic_section_target_is_known
    parsed = {"type": "title", "number": None, "title": "Lukasz Kaiser"}
    docs = ["1. Introduction\nIntro", "2. Methods\nMethods", "Lukasz Kaiser is listed as an author."]
    metas = [
        {"source": "paper.pdf", "section_number": "1", "section_title": "Introduction"},
        {"source": "paper.pdf", "section_number": "2", "section_title": "Methods"},
        {"source": "paper.pdf", "section_number": "", "section_title": ""},
    ]
    assert _generic_section_target_is_known(parsed, docs, metas) is False


def test_generic_named_target_is_section_only_when_heading_exists():
    from backend.rag_pipeline.service import _generic_section_target_is_known
    parsed = {"type": "title", "number": None, "title": "Customer Support Guide"}
    docs = ["9. Customer Support Guide\nSupport body", "10. Data Processing\nOther"]
    metas = [
        {"source": "report.pdf", "section_number": "9", "section_title": "Customer Support Guide"},
        {"source": "report.pdf", "section_number": "10", "section_title": "Data Processing"},
    ]
    assert _generic_section_target_is_known(parsed, docs, metas) is True
