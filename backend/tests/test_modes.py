"""
Automated tests for QueryMind Master Fix:
- Social chat routing & Python control-flow safeguard (Part 2, 14)
- Global structural heading extraction mode (Part 8)
- Named section extraction mode (Part 9)
- Batched full-document summary pipeline (Part 10)
- Groq OCR 429 circuit breaker (Part 18)
- Multi-document source scope filtering (Part 19)
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from backend.rag_pipeline.intent_router import IntentClassification
from backend.rag_pipeline.query import (
    _is_informational_query,
    _is_heading_extraction_query,
    _extract_target_section,
    _extract_document_headings,
    classify_query_intent,
)
from backend.rag_pipeline.service import _answer_question_core
from backend.parsers.document_parser import (
    _extract_vision_from_bytes,
    reset_groq_vision_cooldown,
)


# ===========================================================================
# A. SOCIAL CHAT & PYTHON CONTROL-FLOW SAFEGUARD
# ===========================================================================
class TestPythonControlFlowSafeguard:
    """Ensure genuine social messages go to general chat, but informational
    questions mistakenly classified as casual are overridden to document_query.
    """

    def test_genuine_social_hi(self):
        assert _is_informational_query("hi") is False
        assert _is_informational_query("hello") is False
        assert _is_informational_query("how are you?") is False
        assert _is_informational_query("thanks") is False
        assert _is_informational_query("thank you very much") is False
        assert _is_informational_query("okay") is False

    def test_informational_questions_detected(self):
        assert _is_informational_query("who is Zayn Malik?") is True
        assert _is_informational_query("what is biryani?") is True
        assert _is_informational_query("what is the color of biryani?") is True
        assert _is_informational_query("who is the AI/ML intern?") is True
        assert _is_informational_query("what is Python?") is True
        assert _is_informational_query("explain blockchain") is True
        assert _is_informational_query("tell me about Pakistan") is True
        assert _is_informational_query("who is the founder of X?") is True

    @patch("backend.rag_pipeline.service.generate_general_chat_response")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_safeguard_allows_genuine_social_chat(self, mock_intent, mock_gen_chat):
        mock_intent.return_value = IntentClassification(
            evidence_type="GENERAL_CONVERSATION",
            intent="greeting",
            confidence=0.99,
            reason="User is greeting",
            topic=None,
            question_type="social",
        )
        mock_gen_chat.return_value = "Hello! How can I help you today?"

        res = _answer_question_core("chat_1", "Hi")
        assert res["answer"] == "Hello! How can I help you today?"
        mock_gen_chat.assert_called_once()

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_safeguard_overrides_misclassified_factual_question(self, mock_intent, mock_col):
        # Simulate LLM router mistakenly classifying 'Who is Zayn Malik?' as casual_conversation
        mock_intent.return_value = IntentClassification(
            evidence_type="GENERAL_CONVERSATION",
            intent="casual_conversation",
            confidence=0.85,
            reason="Thought it was casual question",
            topic=None,
            question_type="social",
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        # Document does not contain Zayn Malik
        mock_collection.get.return_value = {
            "documents": ["The company is located in Lahore."],
            "metadatas": [{"source": "company.pdf"}],
            "ids": ["chunk_0"],
        }
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }
        mock_col.return_value = mock_collection

        # Should be overridden to document_query, then blocked by topic gate!
        res = _answer_question_core("chat_1", "Who is Zayn Malik?")
        assert "That information does not appear" in res["answer"]
        assert "company.pdf" in res["answer"]


# ===========================================================================
# B. GLOBAL STRUCTURAL HEADING EXTRACTION (Part 8)
# ===========================================================================
class TestHeadingExtraction:
    """Test dedicated heading extraction mode without RAG truncation."""

    def test_heading_query_intent_detection(self):
        queries = [
            "give me all main headings",
            "list all headings",
            "tell me the main sections",
            "list every section",
            "show all chapter titles",
            "give me the section titles",
            "what are the document headings?",
            "what are the headings?",
            "list all the headings in the document",
        ]
        for q in queries:
            assert _is_heading_extraction_query(q) is True
            assert classify_query_intent(q, [], ["doc.pdf"]) == "heading_extraction"

    def test_candidate_heading_extraction_arbitrary_document(self):
        doc_chunks = [
            "# 1. Executive Summary\nSummary text here.",
            "## 2. Introduction\nIntro text here.",
            "3. System Architecture\nArchitecture details.",
            "4. Data Ingestion Pipeline\nPipeline text.",
            "5. Optical Character Recognition\nOCR details.",
            "6. Text Chunking Strategy\nChunking details.",
            "7. Vector Embeddings\nEmbedding details.",
            "8. Hybrid Retrieval\nRetrieval details.",
            "9. Reranking Module\nReranking details.",
            "10. Intent Classification\nRouter details.",
            "11. Topic Permission Gate\nGate details.",
            "12. Global Structural Extraction\nHeading extraction details.",
            "13. Named Section Extraction\nSection extraction details.",
            "14. Batched Document Summary\nSummary pipeline details.",
            "15. Conclusion and Test Notes\nConclusion text.",
        ]
        headings = _extract_document_headings(doc_chunks)
        assert len(headings) == 15
        assert "15. Conclusion and Test Notes" in headings[-1]
        assert "1. Executive Summary" in headings[0]

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_heading_extraction_returns_all_15_headings(self, mock_intent, mock_col):
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Extract headings",
            topic=None,
            question_type="document_fact",
        )
        doc_chunks = [
            f"{i}. Section Title {i}\nBody content for section {i}."
            for i in range(1, 16)
        ]
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(doc_chunks)
        mock_collection.get.return_value = {
            "documents": doc_chunks,
            "metadatas": [{"source": "report.pdf"} for _ in doc_chunks],
            "ids": [f"chunk_{i}" for i in range(len(doc_chunks))],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Give me all main headings")
        # All 15 section titles must appear in the answer
        for i in range(1, 16):
            assert f"Section Title {i}" in res["answer"]


# ===========================================================================
# C. NAMED SECTION EXTRACTION (Part 9)
# ===========================================================================
class TestSectionExtraction:
    """Test dedicated section extraction mode without context truncation."""

    def test_section_target_extraction(self):
        assert _extract_target_section("tell me conclusion and Test notes") == "conclusion and Test notes"
        assert _extract_target_section("tell me the conclusion") == "conclusion"
        assert _extract_target_section("what is in Executive Overview?") == "Executive Overview"
        assert _extract_target_section("show me section 15") == "15"
        assert _extract_target_section("give me the content under Conclusion and Test Notes") == "Conclusion and Test Notes"
        assert _extract_target_section("explain the Conclusion section") == "Conclusion"
        assert _extract_target_section("tell me everything in section 15") == "15"

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_named_section_extraction_returns_complete_section(self, mock_intent, mock_col):
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Section request",
            topic="Conclusion and Test Notes",
            question_type="document_fact",
        )
        doc_chunks = [
            "14. Previous Section\nDetails of previous section.",
            "15. Conclusion and Test Notes\nFirst paragraph of conclusion.\n\nSecond paragraph of conclusion.",
            "Key finding: All test cases passed with 100% accuracy and zero hallucinations.",
            "16. Next Section\nNext section content.",
        ]
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(doc_chunks)
        mock_collection.get.return_value = {
            "documents": doc_chunks,
            "metadatas": [{"source": "spec.pdf"} for _ in doc_chunks],
            "ids": [f"chunk_{i}" for i in range(len(doc_chunks))],
        }
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Tell me conclusion and Test notes")
        assert "15. Conclusion and Test Notes" in res["answer"]
        assert "Key finding: All test cases passed" in res["answer"]
        # Next section must NOT be included in section 15
        assert "16. Next Section" not in res["answer"]


# ===========================================================================
# D. FULL-DOCUMENT SUMMARY PIPELINE (Part 10)
# ===========================================================================
class TestBatchedSummary:
    """Test full document scope summary."""

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_summary_small_document_single_pass(self, mock_intent, mock_col):
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Summary",
            topic=None,
            question_type="document_fact",
        )
        chunks = [
            "Part 1: Overview of the QueryMind project.",
            "Part 2: Grounding and verification mechanics.",
            "Part 3: Final performance benchmarks.",
        ]
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(chunks)
        mock_collection.get.return_value = {
            "documents": chunks,
            "metadatas": [{"source": "manual.pdf"} for _ in chunks],
            "ids": [f"chunk_{i}" for i in range(len(chunks))],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Summarize the document")
        assert res["answer"] is not None
        assert len(res["sources"]) > 0

    @patch("backend.rag_pipeline.service._langfuse_track_generation")
    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_summary_large_document_batched(self, mock_intent, mock_col, mock_gen):
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Summary",
            topic=None,
            question_type="document_fact",
        )
        # Create a large document (~25,000 chars) that forces multi-batch summarization
        large_chunks = [f"Page {i}: " + ("Significant findings and data point analysis. " * 50) for i in range(1, 15)]
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(large_chunks)
        mock_collection.get.return_value = {
            "documents": large_chunks,
            "metadatas": [{"source": "large_report.pdf"} for _ in large_chunks],
            "ids": [f"chunk_{i}" for i in range(len(large_chunks))],
        }
        mock_col.return_value = mock_collection

        mock_choice = MagicMock()
        mock_choice.message.content = "Summary covering complete document from Page 1 through Page 14."
        mock_gen.return_value.choices = [mock_choice]

        res = _answer_question_core("chat_1", "Summarize this large report")
        # Ensure _langfuse_track_generation was called multiple times (batches + synthesis)
        assert mock_gen.call_count >= 2
        assert "Summary covering complete document" in res["answer"]


# ===========================================================================
# E. GROQ OCR 429 CIRCUIT BREAKER (Part 18)
# ===========================================================================
class TestGroqVisionCircuitBreaker:
    """Test that encountering a 429 rate limit disables Groq Vision and
    routes subsequent calls directly to Gemini Vision.
    """

    def setup_method(self):
        reset_groq_vision_cooldown()

    def teardown_method(self):
        reset_groq_vision_cooldown()

    @patch("backend.parsers.document_parser._gemini_vision_fallback")
    @patch("groq.Groq")
    def test_groq_429_activates_circuit_breaker(self, mock_groq_class, mock_gemini):
        # Configure Groq to raise 429 rate limit error
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Rate limit reached for model qwen/qwen3.8-27b (429 Too Many Requests)"
        )
        mock_groq_class.return_value = mock_client
        mock_gemini.return_value = "Extracted text via Gemini fallback"

        with patch.dict("os.environ", {"GROQ_API_KEY": "fake_key", "GEMINI_API_KEY": "fake_gemini"}):
            # First call: attempts Groq, encounters 429, falls back to Gemini, activates circuit breaker
            res1 = _extract_vision_from_bytes(b"image1", "image/png", "page_1.png")
            assert res1 == "Extracted text via Gemini fallback"
            assert mock_client.chat.completions.create.call_count == 1

            # Second call: circuit breaker is ACTIVE; must NOT attempt Groq at all
            mock_client.chat.completions.create.reset_mock()
            res2 = _extract_vision_from_bytes(b"image2", "image/png", "page_2.png")
            assert res2 == "Extracted text via Gemini fallback"
            # Groq chat.completions.create was NOT called
            assert mock_client.chat.completions.create.call_count == 0


# ===========================================================================
# F. MULTI-DOCUMENT SOURCE SCOPING (Part 19)
# ===========================================================================
class TestSourceScoping:
    """Verify target_source filters document scope in new modes."""

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_selected_source_filters_heading_extraction(self, mock_intent, mock_col):
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Headings",
            topic=None,
            question_type="document_fact",
        )
        docs = [
            "1. Doc A Heading\nContent of Doc A.",
            "1. Doc B Heading\nContent of Doc B.",
        ]
        metas = [{"source": "docA.pdf"}, {"source": "docB.pdf"}]
        ids = ["chunk_0", "chunk_1"]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {"documents": docs, "metadatas": metas, "ids": ids}
        mock_col.return_value = mock_collection

        # Query with target_source="docA.pdf"
        res = _answer_question_core("chat_1", "give me all headings", selected_source="docA.pdf")
        assert "Doc A Heading" in res["answer"]
        assert "Doc B Heading" not in res["answer"]


# ===========================================================================
# G. SECTION U REQUIRED TESTS (FINAL QUERYMIND REPAIR)
# ===========================================================================
class TestSectionURepair:
    """Explicit tests for the 14 scenarios specified in Section U."""

    @pytest.fixture
    def mock_querymind_document(self):
        """Mock document matching the user specification:
        5. Security and Access Rules
        6. Operations Metrics
        15. Conclusion and Test Notes
        """
        doc_chunks = [
            "1. Introduction\nIntro text here.",
            "2. Architecture\nArchitecture text here.",
            "3. Data Flow\nData flow text here.",
            "4. Authentication\nAuth text here.",
            "5. Security and Access Rules\nAll employees must use MFA.\nFirewall rules restrict access to internal VPN.\nRole-based permissions apply to all storage buckets.",
            "6. Operations Metrics\nUptime is 99.99%.\nAverage latency is 120ms.",
            "7. Incident Management\nIncident policies here.",
            "8. Deployment Pipeline\nCI/CD steps here.",
            "9. Monitoring\nMonitoring steps here.",
            "10. Scaling\nScaling steps here.",
            "11. Disaster Recovery\nDR steps here.",
            "12. Audit Logging\nAudit log policies here.",
            "13. Compliance\nSOC2 and GDPR compliance notes.",
            "14. Team Roles\nTeam responsibilities.",
            "15. Conclusion and Test Notes\nFinal conclusions and validation test results.\nZero security vulnerabilities detected.",
        ]
        return doc_chunks

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_u1_tell_me_step_5(self, mock_intent, mock_col, mock_querymind_document):
        """Test 1: 'Tell me step 5' -> Complete content of 5. Security and Access Rules"""
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Step 5",
            topic=None,
            question_type="document_fact",
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(mock_querymind_document)
        mock_collection.get.return_value = {
            "documents": mock_querymind_document,
            "metadatas": [{"source": "manual.pdf"} for _ in mock_querymind_document],
            "ids": [f"chunk_{i}" for i in range(len(mock_querymind_document))],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Tell me step 5")
        assert "Security and Access Rules" in res["answer"]
        assert "All employees must use MFA" in res["answer"]
        assert "Firewall rules restrict access" in res["answer"]
        assert "Operations Metrics" not in res["answer"]

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_u2_tell_me_5_question(self, mock_intent, mock_col, mock_querymind_document):
        """Test 2: 'Tell me 5?' -> Complete content of 5. Security and Access Rules"""
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Tell me 5?",
            topic=None,
            question_type="document_fact",
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(mock_querymind_document)
        mock_collection.get.return_value = {
            "documents": mock_querymind_document,
            "metadatas": [{"source": "manual.pdf"} for _ in mock_querymind_document],
            "ids": [f"chunk_{i}" for i in range(len(mock_querymind_document))],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Tell me 5?")
        assert "Security and Access Rules" in res["answer"]
        assert "All employees must use MFA" in res["answer"]
        assert "Operations Metrics" not in res["answer"]

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_u3_tell_me_section_5(self, mock_intent, mock_col, mock_querymind_document):
        """Test 3: 'Tell me section 5' -> Complete section 5"""
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Section 5",
            topic=None,
            question_type="document_fact",
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(mock_querymind_document)
        mock_collection.get.return_value = {
            "documents": mock_querymind_document,
            "metadatas": [{"source": "manual.pdf"} for _ in mock_querymind_document],
            "ids": [f"chunk_{i}" for i in range(len(mock_querymind_document))],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Tell me section 5")
        assert "Security and Access Rules" in res["answer"]
        assert "All employees must use MFA" in res["answer"]

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_u4_tell_me_about_security_and_access_rules(self, mock_intent, mock_col, mock_querymind_document):
        """Test 4: 'Tell me about Security and Access Rules' -> Complete section 5"""
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Security and Access Rules",
            topic=None,
            question_type="document_fact",
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(mock_querymind_document)
        mock_collection.get.return_value = {
            "documents": mock_querymind_document,
            "metadatas": [{"source": "manual.pdf"} for _ in mock_querymind_document],
            "ids": [f"chunk_{i}" for i in range(len(mock_querymind_document))],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Tell me about Security and Access Rules")
        assert "Security and Access Rules" in res["answer"]
        assert "All employees must use MFA" in res["answer"]
        assert "Operations Metrics" not in res["answer"]

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_u5_u6_all_and_main_headings(self, mock_intent, mock_col, mock_querymind_document):
        """Test 5 & 6: 'Tell me all headings' and 'Give me the main headings' -> ALL 15 headings"""
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Headings",
            topic=None,
            question_type="document_fact",
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(mock_querymind_document)
        mock_collection.get.return_value = {
            "documents": mock_querymind_document,
            "metadatas": [{"source": "manual.pdf"} for _ in mock_querymind_document],
            "ids": [f"chunk_{i}" for i in range(len(mock_querymind_document))],
        }
        mock_col.return_value = mock_collection

        res_all = _answer_question_core("chat_1", "Tell me all headings")
        res_main = _answer_question_core("chat_1", "Give me the main headings")

        for res in [res_all, res_main]:
            assert "Security and Access Rules" in res["answer"]
            assert "Operations Metrics" in res["answer"]
            assert "Conclusion and Test Notes" in res["answer"]

    @patch("backend.rag_pipeline.service.get_chroma_collection")
    @patch("backend.rag_pipeline.service.classify_intent")
    def test_u7_tell_me_conclusion_and_test_notes(self, mock_intent, mock_col, mock_querymind_document):
        """Test 7: 'Tell me conclusion and Test notes' -> Complete 15. Conclusion and Test Notes"""
        mock_intent.return_value = IntentClassification(
            evidence_type="TEXT_EVIDENCE",
            intent="document_query",
            confidence=0.95,
            reason="Conclusion and Test Notes",
            topic=None,
            question_type="document_fact",
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(mock_querymind_document)
        mock_collection.get.return_value = {
            "documents": mock_querymind_document,
            "metadatas": [{"source": "manual.pdf"} for _ in mock_querymind_document],
            "ids": [f"chunk_{i}" for i in range(len(mock_querymind_document))],
        }
        mock_col.return_value = mock_collection

        res = _answer_question_core("chat_1", "Tell me conclusion and Test notes")
        assert "Conclusion and Test Notes" in res["answer"]
        assert "Zero security vulnerabilities detected" in res["answer"]

