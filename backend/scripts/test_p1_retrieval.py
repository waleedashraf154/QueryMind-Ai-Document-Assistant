"""Focused tests for P1 Retrieval Accuracy fixes."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend directory is in sys.path
_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR.parent))
load_dotenv(_BACKEND_DIR / ".env")

from backend.rag_pipeline.vector_store import get_chroma_collection, add_document
from backend.rag_pipeline.common import _chroma_client, RELEVANCE_THRESHOLD, RELEVANCE_THRESHOLD_FALLBACK
from backend.rag_pipeline.retrieval import (
    _detect_target_source,
    _extract_question_keywords,
    _keyword_score,
    _retrieve_relevant_chunks,
    _deduplicate_chunks,
)

def run_tests():
    print("=================================================================")
    print("RUNNING P1 RETRIEVAL ACCURACY TEST SUITE")
    print("=================================================================")

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Source Detection & Single-Doc Locking Fix
    # ─────────────────────────────────────────────────────────────
    print("\n--- TEST 1: Source Detection & Avoid False-Positive Locking ---")
    sources = ["sales_report.txt", "sales_strategy_guide.txt", "financial_summary.pdf"]

    # Loose word 'sales' should NOT lock to either sales file
    q_general = "What are the overall sales figures?"
    target = _detect_target_source(q_general, sources)
    print(f"Query: '{q_general}' -> target: {target}")
    assert target is None, f"Expected None (search all), got '{target}'"

    # Explicit full filename should target the file
    q_explicit = "In sales_report.txt, what was the revenue?"
    target = _detect_target_source(q_explicit, sources)
    print(f"Query: '{q_explicit}' -> target: {target}")
    assert target == "sales_report.txt", f"Expected sales_report.txt, got '{target}'"

    # Ordinal reference should target the first document
    q_ordinal = "Summarize the first document"
    target = _detect_target_source(q_ordinal, sources, document_sources=sources)
    print(f"Query: '{q_ordinal}' -> target: {target}")
    assert target == "sales_report.txt", f"Expected sales_report.txt, got '{target}'"

    # Ordinal reference for 2nd document
    q_ordinal2 = "What does the second document say?"
    target = _detect_target_source(q_ordinal2, sources, document_sources=sources)
    print(f"Query: '{q_ordinal2}' -> target: {target}")
    assert target == "sales_strategy_guide.txt", f"Expected sales_strategy_guide.txt, got '{target}'"
    print("  [PASS] Test 1: Source detection is strict and avoids loose locking.")

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Keywords Extraction (2-letter words, abbreviations, dates)
    # ─────────────────────────────────────────────────────────────
    print("\n--- TEST 2: Keyword Extraction (Names, Dates, Abbreviations) ---")
    kws = _extract_question_keywords("Who is Al and what happened on 12/04/2023?")
    print("Keywords for 'Who is Al and what happened on 12/04/2023?':", kws)
    assert "al" in kws, "2-letter name 'al' must be in keywords"
    assert "12/04/2023" in kws, "date '12/04/2023' must be in keywords"

    kws_abbr = _extract_question_keywords("What is the AI, ML, UI, and ID card info for Jo?")
    print("Keywords for abbreviations and Jo:", kws_abbr)
    assert "ai" in kws_abbr, "abbreviation 'ai' must be in keywords"
    assert "ml" in kws_abbr, "abbreviation 'ml' must be in keywords"
    assert "ui" in kws_abbr, "abbreviation 'ui' must be in keywords"
    assert "id" in kws_abbr, "abbreviation 'id' must be in keywords"
    assert "jo" in kws_abbr, "2-letter name 'jo' must be in keywords"
    print("  [PASS] Test 2: 2-letter names, abbreviations, and dates preserved.")

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Keyword Scoring with Word Boundaries
    # ─────────────────────────────────────────────────────────────
    print("\n--- TEST 3: Keyword Scoring & Boundary Matching ---")
    chunk_with_al = "Manager Al joined the team recently."
    chunk_with_total = "The total general salary for all staff is high."
    score_al_true = _keyword_score(chunk_with_al, ["al"])
    score_al_false = _keyword_score(chunk_with_total, ["al"])
    print(f"Score for 'al' in '{chunk_with_al}': {score_al_true:.3f}")
    print(f"Score for 'al' in '{chunk_with_total}': {score_al_false:.3f}")
    assert score_al_true > 0, "Should match 'Al'"
    assert score_al_false == 0.0, "'al' must NOT match 'total' or 'general'"

    chunk_date = "Project agreement concluded on 12/04/2023 in London."
    score_date = _keyword_score(chunk_date, ["12/04/2023"])
    print(f"Score for '12/04/2023' in date chunk: {score_date:.3f}")
    assert score_date > 0, "Date '12/04/2023' must match"
    print("  [PASS] Test 3: Word boundary scoring prevents substring false positives.")

    # ─────────────────────────────────────────────────────────────
    # TEST 4: Multi-Document Retrieval & Chat Isolation
    # ─────────────────────────────────────────────────────────────
    print("\n--- TEST 4: Multi-Document Search Across All Docs ---")
    chat_multi = "test_chat_multi_doc_p1"
    try:
        add_document(
            chat_multi,
            "Alice is the Principal Architect leading the cloud infrastructure in Berlin.",
            filename="alice_bio.txt",
        )
        add_document(
            chat_multi,
            "Bob is the Lead Security Specialist conducting penetration tests in Madrid.",
            filename="bob_bio.txt",
        )

        coll = get_chroma_collection(chat_multi)
        q_both = "Where do Alice and Bob work?"
        docs, dists, metas = _retrieve_relevant_chunks(
            coll, q_both, doc_count=2, primary_source="alice_bio.txt"
        )
        retrieved_sources = {m.get("source") for m in metas}
        print(f"Query: '{q_both}' -> Retrieved sources: {retrieved_sources}")
        assert "alice_bio.txt" in retrieved_sources, "alice_bio.txt must be retrieved"
        assert "bob_bio.txt" in retrieved_sources, "bob_bio.txt must be retrieved"
        print("  [PASS] Test 4: Both documents retrieved for multi-document query.")
    finally:
        try:
            _chroma_client.delete_collection(get_chroma_collection(chat_multi).name)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # TEST 5: No Artificial Distance Overwrite & Keyword Resiliency
    # ─────────────────────────────────────────────────────────────
    print("\n--- TEST 5: No 0.05 Overwrite & True Hybrid Ranking ---")
    chat_ranking = "test_chat_hybrid_ranking_p1"
    try:
        add_document(
            chat_ranking,
            "Waleed Khan has 6 years of expertise in deep learning, computer vision, and PyTorch.",
            filename="resume_waleed.txt",
        )
        add_document(
            chat_ranking,
            "The candidate likes watching television and playing soccer with friends on weekends.",
            filename="hobbies.txt",
        )

        coll = get_chroma_collection(chat_ranking)
        q_dl = "What is the candidate's deep learning and computer vision background?"
        docs, dists, metas = _retrieve_relevant_chunks(
            coll, q_dl, doc_count=2, primary_source="resume_waleed.txt"
        )
        print("Retrieved chunks:")
        for d, dist, m in zip(docs, dists, metas):
            print(f"  dist={dist:.4f} source={m.get('source')} | {d[:60]}")

        assert len(docs) >= 1
        assert "deep learning" in docs[0].lower()
        # Ensure distances are not artificially forced to 0.05
        assert dists[0] > 0.06, f"Distance {dists[0]} should be the true vector distance, not overwritten to 0.05"
        print("  [PASS] Test 5: True vector distances preserved, relevant chunk ranked #1.")
    finally:
        try:
            _chroma_client.delete_collection(get_chroma_collection(chat_ranking).name)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # TEST 6: Tightened Fallback Threshold (0.78) Rejects Unrelated Queries
    # ─────────────────────────────────────────────────────────────
    print("\n--- TEST 6: Unrelated Query Rejection (Fallback 0.78) ---")
    chat_unrelated = "test_chat_unrelated_p1"
    try:
        add_document(
            chat_unrelated,
            "Alexander Wright is a corporate tax accountant who specializes in IRS audits and payroll taxes.",
            filename="tax_profile.txt",
        )

        coll = get_chroma_collection(chat_unrelated)
        q_cake = "What is the secret recipe for baking homemade chocolate cheesecake?"
        docs, dists, metas = _retrieve_relevant_chunks(
            coll, q_cake, doc_count=1, primary_source="tax_profile.txt"
        )
        print(f"Query: '{q_cake}' -> Retrieved {len(docs)} chunks")
        assert len(docs) == 0, f"Unrelated query should retrieve 0 chunks under 0.78 threshold, got {len(docs)}"
        print("  [PASS] Test 6: Unrelated query correctly rejected by 0.78 threshold.")
    finally:
        try:
            _chroma_client.delete_collection(get_chroma_collection(chat_unrelated).name)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # TEST 7: Deduplication of Duplicate and Near-Duplicate Chunks
    # ─────────────────────────────────────────────────────────────
    print("\n--- TEST 7: Exact & Near-Duplicate Deduplication ---")
    raw_d = [
        "Machine learning algorithms require substantial training data and computational resources.",
        "Machine learning algorithms require substantial training data and computational resources.",  # Exact duplicate
        "Machine learning algorithms require substantial training data and computational resources today.",  # Near-duplicate (Jaccard = 0.909 >= 0.85)
        "Database indexing dramatically accelerates query response times on large tables.",
    ]
    raw_dist = [0.30, 0.30, 0.32, 0.40]
    raw_m = [{"s": 1}, {"s": 2}, {"s": 3}, {"s": 4}]

    k_docs, k_dists, k_metas = _deduplicate_chunks(raw_d, raw_dist, raw_m)
    print(f"Original: {len(raw_d)} chunks -> After dedup: {len(k_docs)} chunks")
    assert len(k_docs) == 2, f"Expected 2 unique chunks after dedup, got {len(k_docs)}"
    print("  [PASS] Test 7: Exact and near-duplicates successfully filtered.")

    print("\n=================================================================")
    print("ALL P1 RETRIEVAL ACCURACY TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
