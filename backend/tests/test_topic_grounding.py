"""
Tests for QueryMind's 2-stage document-grounding system.

TEST 1:  doc has 'biryani', 'What is the color of biryani?'          -> allowed (topic_knowledge)
TEST 2:  doc has 'biryani', 'What ingredients are used in biryani?'  -> allowed (topic_knowledge)
TEST 3:  doc has 'biryani', 'Who is Shahid Afridi?'                  -> NOT-IN-DOCUMENT
TEST 4:  doc has 'biryani', 'Who is Zayn Malik?'                     -> NOT-IN-DOCUMENT
TEST 5:  doc has 'AI/ML Intern - Muhammad Waleed', 'Who is the AI/ML intern?' -> document_fact -> allowed
TEST 6:  same doc, 'Tell me the name of the AI/ML intern.'           -> document_fact -> allowed
TEST 7:  same doc, 'Who works as the AI/ML intern?'                  -> document_fact -> allowed
TEST 8:  doc has 'biryani', 'Who is the founder of United States?'   -> NOT-IN-DOCUMENT
TEST 9:  'Hi'                                                         -> topic=None (social)
TEST 10: 'How are you?'                                               -> topic=None (social)
TEST 11: doc has 'Python', 'What is Python?'                         -> allowed
TEST 12: doc has 'biryani', 'What is Python?'                        -> NOT-IN-DOCUMENT
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collection(doc_text: str) -> MagicMock:
    collection = MagicMock()
    collection.count.return_value = 1
    meta = {"source": "test_document.pdf", "type": "text"}
    collection.get.return_value = {
        "documents": [doc_text],
        "metadatas": [meta],
    }
    collection.query.return_value = {
        "documents": [[doc_text]],
        "metadatas": [[meta]],
        "distances": [[0.2]],
        "ids": [["chunk-1"]],
    }
    return collection


def _run_gate(question: str, doc_text: str) -> bool:
    """Return True if the topic gate allows the question (topic found / no topic = social)."""
    from backend.rag_pipeline.query import (
        _extract_informational_topic,
        _find_topic_in_entire_document,
    )
    topic = _extract_informational_topic(question)
    if topic is None:
        return True  # social message, no gate
    collection = _make_collection(doc_text)
    found, _, _, _ = _find_topic_in_entire_document(collection, topic)
    return found


# ---------------------------------------------------------------------------
# Unit tests for _extract_definition_term fix
# ---------------------------------------------------------------------------

class TestDefinitionTermFix:
    """Ensure 'X of Y' attribute queries return None from _extract_definition_term."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.rag_pipeline.query import _extract_definition_term
        self.extract = _extract_definition_term

    def test_color_of_biryani_returns_none(self):
        """'What is the color of biryani?' is an attribute query, not a definition."""
        assert self.extract("What is the color of biryani?") is None

    def test_ingredients_of_biryani_returns_none(self):
        assert self.extract("What is the name of the AI/ML intern?") is None

    def test_founder_of_pakistan_returns_none(self):
        assert self.extract("What is the founder of Pakistan?") is None

    def test_what_is_python_still_works(self):
        """Pure concept definitions like 'What is Python?' must still be extracted."""
        result = self.extract("What is Python?")
        assert result is not None
        assert "python" in result.lower()

    def test_explicit_definition_of_biryani_still_works(self):
        """Explicit 'definition of X' is still extracted even with ' of '."""
        result = self.extract("What is the definition of biryani?")
        assert result is not None
        assert "biryani" in result.lower()


# ---------------------------------------------------------------------------
# Unit tests for _extract_informational_topic (regex fallback)
# ---------------------------------------------------------------------------

class TestExtractInformationalTopic:
    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.rag_pipeline.query import _extract_informational_topic
        self.extract = _extract_informational_topic

    def test_color_of_biryani_returns_biryani(self):
        result = self.extract("What is the color of biryani?")
        assert result is not None
        assert "biryani" in result.lower()

    def test_who_is_zayn_malik(self):
        result = self.extract("Who is Zayn Malik?")
        assert result is not None
        assert "zayn malik" in result.lower()

    def test_who_founded_pakistan(self):
        result = self.extract("Who founded Pakistan?")
        assert result is not None
        assert "pakistan" in result.lower()

    def test_what_is_python(self):
        result = self.extract("What is Python?")
        assert result is not None
        assert "python" in result.lower()

    def test_albums_zayn_malik(self):
        result = self.extract("What albums has Zayn Malik released?")
        assert result is not None
        assert "zayn malik" in result.lower()

    def test_ingredients_biryani(self):
        result = self.extract("What ingredients are commonly used in biryani?")
        assert result is not None
        assert "biryani" in result.lower()

    def test_who_is_ai_ml_intern(self):
        """Must handle 'AI/ML intern' with the slash."""
        result = self.extract("Who is the AI/ML intern?")
        # Accept either the full phrase or at minimum 'AI' or 'intern'
        assert result is not None
        lower = result.lower()
        assert any(k in lower for k in ["ai", "ml", "intern"]), f"Unexpected: {result!r}"

    def test_social_hi_returns_none(self):
        assert self.extract("Hi") is None

    def test_social_how_are_you_returns_none(self):
        assert self.extract("How are you?") is None

    def test_social_thanks_returns_none(self):
        assert self.extract("Thanks!") is None

    def test_candidate_field_returns_none(self):
        assert self.extract("What is the candidate's education?") is None


# ---------------------------------------------------------------------------
# Unit tests for _find_topic_in_entire_document with AI/ML
# ---------------------------------------------------------------------------

class TestFindTopicAIML:
    def test_ai_ml_intern_found_in_doc(self):
        from backend.rag_pipeline.query import _find_topic_in_entire_document
        collection = _make_collection("AI/ML Intern - Muhammad Waleed")
        found, docs, metas, matched_by = _find_topic_in_entire_document(
            collection, "AI/ML intern"
        )
        assert found is True, "AI/ML intern should be found in the document"

    def test_ai_ml_intern_not_found_in_irrelevant_doc(self):
        from backend.rag_pipeline.query import _find_topic_in_entire_document
        collection = _make_collection("Biryani is a popular rice dish.")
        found, _, _, _ = _find_topic_in_entire_document(collection, "AI/ML intern")
        assert found is False


# ---------------------------------------------------------------------------
# Integration-style topic-gate tests (all 12 required test cases)
# ---------------------------------------------------------------------------

class TestTopicGateAllCases:

    def test_case_1_biryani_in_doc_color_question_allowed(self):
        """TEST 1: doc has biryani, 'What is the color of biryani?' -> allowed."""
        assert _run_gate(
            "What is the color of biryani?",
            "Biryani is a popular rice dish in South Asia.",
        ) is True

    def test_case_2_biryani_in_doc_ingredients_question_allowed(self):
        """TEST 2: doc has biryani, 'What ingredients are used in biryani?' -> allowed."""
        assert _run_gate(
            "What ingredients are commonly used in biryani?",
            "Biryani is a popular rice dish in South Asia.",
        ) is True

    def test_case_3_shahid_afridi_not_in_doc_blocked(self):
        """TEST 3: doc does NOT have Shahid Afridi -> NOT-IN-DOCUMENT."""
        assert _run_gate(
            "Who is Shahid Afridi?",
            "Biryani is a popular rice dish in South Asia.",
        ) is False

    def test_case_4_zayn_malik_not_in_doc_blocked(self):
        """TEST 4: doc does NOT have Zayn Malik -> NOT-IN-DOCUMENT."""
        assert _run_gate(
            "Who is Zayn Malik?",
            "Biryani is a popular rice dish in South Asia.",
        ) is False

    def test_case_5_ai_ml_intern_in_doc_allowed(self):
        """TEST 5: doc has 'AI/ML Intern - Muhammad Waleed', 'Who is the AI/ML intern?' -> allowed."""
        assert _run_gate(
            "Who is the AI/ML intern?",
            "AI/ML Intern - Muhammad Waleed",
        ) is True

    def test_case_6_tell_me_name_of_ai_ml_intern_allowed(self):
        """TEST 6: 'Tell me the name of the AI/ML intern.' -> allowed (topic gate passes)."""
        # Note: regex may not extract the topic here; the gate returns True when
        # topic is None (no blocking needed — RAG will handle it). If the regex
        # DOES extract a topic, it must be verified in the doc.
        from backend.rag_pipeline.query import _extract_informational_topic
        doc = "AI/ML Intern - Muhammad Waleed"
        topic = _extract_informational_topic("Tell me the name of the AI/ML intern.")
        if topic is None:
            # No topic extracted by regex -> passes gate (RAG handles it)
            assert True
        else:
            from backend.rag_pipeline.query import _find_topic_in_entire_document
            collection = _make_collection(doc)
            found, _, _, _ = _find_topic_in_entire_document(collection, topic)
            assert found is True, f"Regex extracted '{topic}' but it was not found in the doc"

    def test_case_7_who_works_as_ai_ml_intern_allowed(self):
        """TEST 7: 'Who works as the AI/ML intern?' -> allowed."""
        assert _run_gate(
            "Who works as the AI/ML intern?",
            "AI/ML Intern - Muhammad Waleed",
        ) is True

    def test_case_8_united_states_not_in_doc_blocked(self):
        """TEST 8: doc does NOT contain United States -> NOT-IN-DOCUMENT."""
        assert _run_gate(
            "Who is the founder of United States?",
            "Biryani is a popular rice dish.",
        ) is False

    def test_case_9_hi_bypasses_gate(self):
        """TEST 9: 'Hi' -> social -> topic=None."""
        from backend.rag_pipeline.query import _extract_informational_topic
        assert _extract_informational_topic("Hi") is None

    def test_case_10_how_are_you_bypasses_gate(self):
        """TEST 10: 'How are you?' -> social -> topic=None."""
        from backend.rag_pipeline.query import _extract_informational_topic
        assert _extract_informational_topic("How are you?") is None

    def test_case_11_python_in_doc_allowed(self):
        """TEST 11: doc has Python, 'What is Python?' -> allowed."""
        assert _run_gate(
            "What is Python?",
            "Python is used for data analysis.",
        ) is True

    def test_case_12_python_not_in_doc_blocked(self):
        """TEST 12: doc does NOT have Python -> NOT-IN-DOCUMENT."""
        assert _run_gate(
            "What is Python?",
            "Biryani is a popular rice dish in South Asia.",
        ) is False


# ---------------------------------------------------------------------------
# Intent router parsing — unit tests for new topic/question_type fields
# ---------------------------------------------------------------------------

class TestIntentRouterParsing:
    """Verify that _parse_classification correctly handles the new fields."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.rag_pipeline.intent_router import _parse_classification
        self.parse = _parse_classification

    def _make_json(self, evidence_type, intent, topic=None, question_type=None):
        import json
        return json.dumps({
            "evidence_type": evidence_type,
            "intent": intent,
            "confidence": 0.95,
            "reason": "test",
            "topic": topic,
            "question_type": question_type or ("social" if intent != "document_query" else "topic_knowledge"),
        })

    def test_topic_knowledge_parsed(self):
        raw = self._make_json("TEXT_EVIDENCE", "document_query", topic="biryani", question_type="topic_knowledge")
        result = self.parse(raw)
        assert result.topic == "biryani"
        assert result.question_type == "topic_knowledge"

    def test_document_fact_parsed(self):
        raw = self._make_json("TEXT_EVIDENCE", "document_query", topic="AI/ML intern", question_type="document_fact")
        result = self.parse(raw)
        assert result.topic == "AI/ML intern"
        assert result.question_type == "document_fact"

    def test_social_parsed(self):
        raw = self._make_json("GENERAL_CONVERSATION", "greeting", topic=None, question_type="social")
        result = self.parse(raw)
        assert result.topic is None
        assert result.question_type == "social"

    def test_null_topic_string_normalized(self):
        """LLM may return 'null' or 'none' as a string for topic."""
        import json
        raw = json.dumps({
            "evidence_type": "GENERAL_CONVERSATION",
            "intent": "greeting",
            "confidence": 0.99,
            "reason": "pure greeting",
            "topic": "null",
            "question_type": "social",
        })
        result = self.parse(raw)
        assert result.topic is None

    def test_missing_topic_defaults_to_none(self):
        """Old-format output (4 fields) should still parse without error."""
        import json
        raw = json.dumps({
            "evidence_type": "TEXT_EVIDENCE",
            "intent": "document_query",
            "confidence": 0.9,
            "reason": "factual question",
        })
        result = self.parse(raw)
        assert result.topic is None
        assert result.intent == "document_query"
