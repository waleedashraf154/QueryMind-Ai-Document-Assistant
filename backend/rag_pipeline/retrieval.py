from __future__ import annotations
import os
import re
import sys
import time
from typing import Optional

from .common import RELEVANCE_THRESHOLD, RELEVANCE_THRESHOLD_FALLBACK, NEAR_DUPLICATE_JACCARD, TOP_K, RERANK_TOP_N, MAX_CONTEXT_CHARS, _STOP_WORDS, _FIELD_ALIASES
from .embeddings import embed_texts
from .query import (
    _safe_source_list,
    _is_image_content_query,
    _is_plural_image_content_query,
    _extract_structured_references,
    is_image_file,
)
from .chunking import get_adaptive_threshold
from backend.core.config import STORAGE_DIR
from backend.parsers.document_parser import get_or_render_page_image

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace — used for exact-duplicate detection."""
    return re.sub(r"\s+", " ", text.strip().lower())

def _word_set(text: str) -> set[str]:
    """Return the set of words in *text* for Jaccard similarity."""
    return set(re.findall(r"\w+", text.lower()))

def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings. Range [0, 1]."""
    sa, sb = _word_set(a), _word_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _deduplicate_chunks(
    docs: list[str],
    distances: list[float],
    metas: list[dict],
) -> tuple[list[str], list[float], list[dict]]:
    """
    Remove exact duplicates and near-duplicates from retrieved chunks.

    The first occurrence of each unique chunk is kept (chunks are already
    sorted by relevance distance from ChromaDB).
    """
    kept_docs: list[str] = []
    kept_distances: list[float] = []
    kept_metas: list[dict] = []
    seen_normalised: set[str] = set()

    for doc, dist, meta in zip(docs, distances, metas):
        norm = _normalise(doc)

        if norm in seen_normalised:
            print(f"[rag] DEDUP: exact duplicate removed (dist={dist:.3f})", file=sys.stderr)
            continue

        is_near_dup = False
        for kept in kept_docs:
            if _jaccard(doc, kept) >= NEAR_DUPLICATE_JACCARD:
                is_near_dup = True
                print(
                    f"[rag] DEDUP: near-duplicate removed "
                    f"(jaccard≥{NEAR_DUPLICATE_JACCARD}, dist={dist:.3f})",
                    file=sys.stderr,
                )
                break

        if is_near_dup:
            continue

        seen_normalised.add(norm)
        kept_docs.append(doc)
        kept_distances.append(dist)
        kept_metas.append(meta)

    return kept_docs, kept_distances, kept_metas

def _extract_question_keywords(question: str) -> list[str]:
    """
    Extract meaningful keywords from *question*, removing stop-words.

    Preserves exact dates (e.g. 12/04/2023), 2-letter names (e.g. 'Al', 'Jo'),
    and abbreviations (e.g. 'AI', 'ML', 'UI', 'ID') using word-boundary matching.
    Also expands aliases and singular/plural variations so queries like
    "What languages does the candidate know?" reliably match "LANGUAGE".

    Returns a list of lowercase strings, most important first.
    """
    # 1. Extract explicit date strings first (e.g. 12/04/2023, 2023-04-12)
    dates = re.findall(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b", question)

    # 2. Extract words
    words = re.findall(r"\b\w+\b", question.lower())

    # Keep dates and words with len >= 2 that are not stop-words
    keywords: list[str] = [d.lower() for d in dates]
    for w in words:
        if w not in _STOP_WORDS and len(w) >= 2 and w not in keywords:
            keywords.append(w)

    # Fallback: if stop-word removal left nothing, use all words len>=2
    if not keywords:
        keywords = [w for w in words if len(w) >= 2]

    # Expand with singular/plural and field aliases (skip dates/numbers)
    expanded: list[str] = list(keywords)
    for kw in list(keywords):
        if re.search(r"\d", kw):
            continue
        if kw.endswith("s") and len(kw) > 3 and kw[:-1] not in expanded:
            expanded.append(kw[:-1])
        elif not kw.endswith("s") and len(kw) >= 2 and (kw + "s") not in expanded:
            expanded.append(kw + "s")
        if kw in _FIELD_ALIASES:
            for alias in _FIELD_ALIASES[kw]:
                if alias not in expanded:
                    expanded.append(alias)

    return expanded

def _keyword_score(chunk: str, keywords: list[str]) -> float:
    """
    Score a chunk by how many question keywords it contains.

    Returns a value in [0, 1] where 1.0 = all keywords present.
    Uses whole-word boundary matching so 'al' doesn't match 'total' and
    'jo' doesn't match 'major'.
    Score is weighted so earlier (more important) keywords count more.
    """
    if not keywords:
        return 0.0

    chunk_lower = chunk.lower()
    total_weight = 0.0
    matched_weight = 0.0

    for rank, kw in enumerate(keywords):
        kw_l = kw.lower()
        # Earlier keywords carry more weight (exponential decay)
        weight = 1.0 / (1.0 + rank * 0.3)
        total_weight += weight

        if re.search(r"[^a-zA-Z0-9]", kw_l):
            matched = bool(re.search(rf"(?:^|[\s,;:(<\[]){re.escape(kw_l)}(?:$|[\s,;:)>\]\.\?!])", chunk_lower)) or kw_l in chunk_lower
        else:
            matched = bool(re.search(rf"\b{re.escape(kw_l)}\b", chunk_lower))

        if matched:
            matched_weight += weight

    return matched_weight / total_weight if total_weight > 0 else 0.0

def _rerank_chunks(
    docs: list[str],
    distances: list[float],
    metas: list[dict],
    question: str,
    top_n: int,
) -> tuple[list[str], list[float], list[dict]]:
    """
    Rerank *docs* by blending ChromaDB vector distance with a keyword
    overlap score derived from *question*.

    Blended score formula (lower = better, consistent with cosine distance):
        blended = α × vector_distance − β × keyword_score

    where:
        α = 0.55  (vector similarity weight)
        β = 0.45  (keyword overlap weight)

    This means a chunk that is highly semantically relevant AND contains
    the exact question keywords will rank first.

    For the "intermediate education" example:
    - "Intermediate — ICS-PHY — Govt. Islamia College Civil Lines" matches
      all keywords → keyword_score ≈ 1.0 → blended score is heavily lowered
    - "Matric — Computer Science — PAK Angels Grammar School" matches 0
      keywords → blended score stays near the raw vector distance

    Parameters
    ----------
    docs, distances, metas : matched lists from ChromaDB / deduplication
    question               : original user question
    top_n                  : how many chunks to return

    Returns
    -------
    (reranked_docs, reranked_distances, reranked_metas) — truncated to top_n
    """
    keywords = _extract_question_keywords(question)

    if not keywords:
        # No keywords extracted — return as-is, truncated to top_n
        return docs[:top_n], distances[:top_n], metas[:top_n]

    alpha = 0.55  # vector distance weight
    beta  = 0.45  # keyword score weight

    scored: list[tuple[float, int]] = []
    for idx, (doc, dist) in enumerate(zip(docs, distances)):
        ks = _keyword_score(doc, keywords)
        blended = alpha * dist - beta * ks
        scored.append((blended, idx))
        print(
            f"[rag] RERANK [{idx+1}] dist={dist:.3f} kw={ks:.3f} blended={blended:.3f} "
            f"| preview: {doc[:80].replace(chr(10), ' ')}",
            file=sys.stderr,
        )

    scored.sort(key=lambda t: t[0])  # ascending blended score → best first

    top_indices = [idx for _, idx in scored[:top_n]]

    reranked_docs      = [docs[i]      for i in top_indices]
    reranked_distances = [distances[i] for i in top_indices]
    reranked_metas     = [metas[i]     for i in top_indices]

    print(
        f"[rag] RERANK: top-{top_n} after reranking: "
        f"{[round(distances[i], 3) for i in top_indices]}",
        file=sys.stderr,
    )

    return reranked_docs, reranked_distances, reranked_metas

def _literal_keyword_matches(
    collection,
    keywords: list[str],
    already_seen: set[str],
    question_embedding: Optional[list[float]] = None,
    target_source: Optional[str] = None,
) -> tuple[list[str], list[float], list[dict]]:
    """
    Scan the document collection for chunks that literally contain substantive keywords.

    Computes true cosine vector distance against question_embedding rather than
    assigning an artificial 0.05.
    """
    if not keywords:
        return [], [], []

    try:
        all_data = collection.get(include=["documents", "metadatas", "embeddings"])
    except Exception as exc:
        print(f"[rag] literal keyword scan failed to fetch collection: {exc}", file=sys.stderr)
        return [], [], []

    raw_docs = all_data.get("documents")
    all_docs: list[str] = list(raw_docs) if raw_docs is not None else []
    raw_metas = all_data.get("metadatas")
    all_metas: list[dict] = list(raw_metas) if raw_metas is not None else []
    raw_embeddings = all_data.get("embeddings")
    all_embeddings = raw_embeddings if raw_embeddings is not None else []

    specific_keywords = [
        kw for kw in keywords
        if len(kw) >= 2 and kw not in {
            "candidate", "person", "information", "document", "tell", "tells",
            "telling", "told", "details", "what", "where", "when", "about",
            "give", "gives", "show", "shows", "state", "states", "describe",
            "describes", "mention", "mentions", "find", "finds",
        }
    ]
    patterns = []
    for kw in specific_keywords:
        kw_l = kw.lower()
        if re.search(r"[^a-zA-Z0-9]", kw_l):
            patterns.append(re.compile(rf"(?:^|[\s,;:(<\[]){re.escape(kw_l)}(?:$|[\s,;:)>\]\.\?!])", re.IGNORECASE))
        else:
            patterns.append(re.compile(rf"\b{re.escape(kw_l)}\b", re.IGNORECASE))

    if not patterns:
        return [], [], []

    matched_docs: list[str] = []
    matched_distances: list[float] = []
    matched_metas: list[dict] = []

    for i, doc in enumerate(all_docs):
        meta = all_metas[i] if i < len(all_metas) else {}
        if target_source and (meta or {}).get("source") != target_source:
            continue
        if doc in already_seen:
            continue
        if any(p.search(doc) for p in patterns):
            matched_docs.append(doc)
            chunk_emb = all_embeddings[i] if i < len(all_embeddings) else None
            if question_embedding is not None and chunk_emb is not None and len(chunk_emb) > 0:
                dot_prod = sum(q * c for q, c in zip(question_embedding, chunk_emb))
                real_dist = max(0.0, min(2.0, 1.0 - dot_prod))
            else:
                real_dist = 0.70
            matched_distances.append(real_dist)
            matched_metas.append(meta or {})
            print(
                f"[rag] LITERAL KEYWORD HIT (true dist={real_dist:.4f}): "
                f"{doc[:80].replace(chr(10), ' ')}",
                file=sys.stderr,
            )

    return matched_docs, matched_distances, matched_metas

def _latest_image_source(
    collection,
    chat_id: str,
    target_source: Optional[str] = None,
) -> Optional[str]:
    """Return the most recently stored image source for a chat."""
    try:
        data = collection.get(include=["metadatas"])
        metas = data.get("metadatas", []) or []
    except Exception as exc:
        print(f"[rag] latest-image lookup failed: {exc}", file=sys.stderr)
        return target_source

    image_sources = []
    seen = set()
    for meta in metas:
        meta = meta or {}
        if meta.get("type") != "image":
            continue
        source = str(meta.get("source", "")).strip()
        if not source or source in seen:
            continue
        if target_source and source != target_source and os.path.basename(source) != target_source:
            continue
        seen.add(source)
        image_sources.append(source)

    if not image_sources:
        return target_source

    # Chroma preserves the insertion order returned by this collection read
    # for the current chat. The last unique image source is therefore treated
    # as the most recently uploaded image. This avoids coupling retrieval to
    # backend storage paths and prevents undefined _BACKEND_DIR errors.
    return image_sources[-1]



def _retrieve_named_section(
    collection,
    target_source: Optional[str] = None,
    section_number: Optional[str] = None,
    section_title: Optional[str] = None,
) -> tuple[list[str], list[float], list[dict]]:
    """Retrieve every chunk belonging to one named document section in source order.

    New indexes use section metadata. Older indexes are handled by a deterministic
    fallback that detects a numbered heading at the beginning of a chunk and collects
    contiguous chunks until the next numbered heading.
    """
    try:
        data = collection.get(include=["documents", "metadatas"])
        docs = list(data.get("documents") or [])
        metas = list(data.get("metadatas") or [])
        ids = list(data.get("ids") or [])
    except Exception as exc:
        print(f"[rag] named-section retrieval failed: {exc}", file=sys.stderr)
        return [], [], []

    def norm(v: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (v or "").casefold()).strip()

    wanted_title = norm(section_title or "")

    raw_rows = []
    for idx, doc in enumerate(docs):
        meta = metas[idx] if idx < len(metas) else {}
        meta = meta or {}
        if target_source and os.path.basename(str(meta.get("source", ""))).casefold() != os.path.basename(target_source).casefold():
            continue
        chunk_idx = meta.get("chunk_index")
        if isinstance(chunk_idx, int):
            order = (0, chunk_idx)
        else:
            cid = str(ids[idx]) if idx < len(ids) else str(idx)
            m = re.search(r"_(\d+)$", cid)
            order = (1, int(m.group(1))) if m else (2, idx)
        raw_rows.append((order, doc or "", meta, idx))

    raw_rows.sort(key=lambda r: r[0])

    # 1) Preferred path: exact section metadata. For a hierarchical parent such
    # as 3.2, include all descendants (3.2.1, 3.2.2, ...) in source order.
    # This is what a human means by asking for the contents of a parent section.
    selected_rows = []
    wanted_num = str(section_number or "").strip()
    for order, doc, meta, idx in raw_rows:
        match = False
        stored_num = str(meta.get("section_number", "") or "").strip()
        if wanted_num:
            match = stored_num == wanted_num or stored_num.startswith(wanted_num + ".")
        if not match and wanted_title:
            stored_title = norm(str(meta.get("section_title", "")))
            match = stored_title == wanted_title or wanted_title in stored_title
        if match:
            selected_rows.append((order, doc, meta))

    # 2) Legacy fallback: section metadata may be absent on existing collections.
    if not selected_rows and (section_number or wanted_title):
        number_heading = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+(.+?)\s*$")
        anchor_idx: Optional[int] = None
        anchor_number: Optional[str] = None
        for i, (order, doc, meta, original_idx) in enumerate(raw_rows):
            first_line = (doc or "").splitlines()[0].strip() if (doc or "").splitlines() else ""
            m = number_heading.match(first_line)
            if not m:
                continue
            num, title = m.group(1), m.group(2).strip()
            title_norm = norm(title)
            if section_number and num == str(section_number):
                anchor_idx, anchor_number = i, num
                break
            if wanted_title and (title_norm == wanted_title or wanted_title in title_norm or title_norm in wanted_title):
                anchor_idx, anchor_number = i, num
                break

        if anchor_idx is not None:
            parent_depth = len(anchor_number.split(".")) if anchor_number else 1
            for i in range(anchor_idx, len(raw_rows)):
                order, doc, meta, original_idx = raw_rows[i]
                first_line = (doc or "").splitlines()[0].strip() if (doc or "").splitlines() else ""
                m = number_heading.match(first_line)
                if i > anchor_idx and m:
                    candidate_num = m.group(1)
                    # Keep the requested section and all descendants; stop at
                    # the next sibling/ancestor heading.
                    if not (candidate_num == anchor_number or candidate_num.startswith(anchor_number + ".")):
                        if len(candidate_num.split(".")) <= parent_depth:
                            break
                        # A deeper unrelated structure should not occur in a
                        # well-formed outline; conservatively stop here.
                        break
                selected_rows.append((order, doc, meta))

    # 3) Resume/CV fallback for older indexes without section metadata.
    # Match explicit "Section: PROFESSIONAL EXPERIENCE" prefixes and collect all
    # contiguous chunks carrying the same section prefix until the next section.
    if not selected_rows and wanted_title:
        for i, (order, doc, meta, original_idx) in enumerate(raw_rows):
            first_line = (doc or "").splitlines()[0].strip() if (doc or "").splitlines() else ""
            m = re.match(r"^Section:\s*(.+?)\s*$", first_line, re.IGNORECASE)
            if not m:
                continue
            title_norm = norm(m.group(1))
            if not (title_norm == wanted_title or wanted_title in title_norm or title_norm in wanted_title):
                continue
            selected_rows.append((order, doc, meta))
            for j in range(i + 1, len(raw_rows)):
                order2, doc2, meta2, original_idx2 = raw_rows[j]
                first2 = (doc2 or "").splitlines()[0].strip() if (doc2 or "").splitlines() else ""
                m2 = re.match(r"^Section:\s*(.+?)\s*$", first2, re.IGNORECASE)
                if not m2:
                    selected_rows.append((order2, doc2, meta2))
                    continue
                title2_norm = norm(m2.group(1))
                if title2_norm == title_norm:
                    selected_rows.append((order2, doc2, meta2))
                    continue
                break
            break

    selected_docs: list[str] = []
    selected_metas: list[dict] = []
    seen_text: set[str] = set()
    for _, doc, meta in sorted(selected_rows, key=lambda r: r[0]):
        cleaned = (doc or "").strip()
        if not cleaned:
            continue
        key = _normalise(cleaned)
        if key in seen_text:
            continue
        seen_text.add(key)
        selected_docs.append(cleaned)
        selected_metas.append(meta or {})

    print(
        f"[rag] named-section retrieval: source={target_source!r} number={section_number!r} "
        f"title={section_title!r} chunks={len(selected_docs)}",
        file=sys.stderr,
    )
    return selected_docs, [0.0] * len(selected_docs), selected_metas


def _retrieve_structured_sections(
    collection,
    references: list[tuple[str, str]],
    target_source: Optional[str] = None,
    max_chunks_per_reference: int = 40,
) -> tuple[list[str], list[float], list[dict]]:
    """Retrieve complete sections referenced by the user.

    This is document-agnostic. It handles questions, chapters, sections, steps,
    problems, exercises, examples, tasks, items, pages, and parts. For numbered
    structures, retrieval follows the stored chunk order until the next heading
    of the same type, preserving multi-page/scanned sections.
    """
    if not references:
        return [], [], []

    try:
        data = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"[rag] structured-section scan failed: {exc}", file=sys.stderr)
        return [], [], []

    docs = list(data.get("documents") or [])
    metas = list(data.get("metadatas") or [])
    ids = list(data.get("ids") or [])

    rows = []
    for idx, doc in enumerate(docs):
        meta = metas[idx] if idx < len(metas) else {}
        if target_source and (meta or {}).get("source") != target_source:
            continue
        chunk_id = str(ids[idx]) if idx < len(ids) else str(idx)
        rows.append((idx, chunk_id, doc or "", meta or {}))

    def _order_key(row):
        original_idx, chunk_id, _, _ = row
        match = re.search(r"_(\d+)$", chunk_id)
        if match:
            return (0, int(match.group(1)))
        return (1, original_idx)

    rows.sort(key=_order_key)

    structured_num = r"(\d+(?:\.\d+)*)"
    heading_patterns = {
        "question": re.compile(rf"(?i)(?<!\w)(?:q|question)\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "chapter": re.compile(rf"(?i)(?<!\w)(?:chapter|chap\.)\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "section": re.compile(rf"(?i)(?<!\w)(?:section|sec\.?|subsection|sub-section)\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "step": re.compile(rf"(?i)(?<!\w)step\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "point": re.compile(rf"(?i)(?<!\w)point\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "heading": re.compile(rf"(?i)(?<!\w)heading\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "problem": re.compile(rf"(?i)(?<!\w)(?:problem|prob\.)\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "exercise": re.compile(rf"(?i)(?<!\w)exercise\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "example": re.compile(rf"(?i)(?<!\w)example\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "task": re.compile(rf"(?i)(?<!\w)task\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "item": re.compile(rf"(?i)(?<!\w)item\s*(?:number|no\.?|#)?\s*{structured_num}(?![\d.])"),
        "part": re.compile(r"(?i)(?<!\w)part\s*([A-Za-z])(?![A-Za-z])"),
    }

    selected: list[tuple[str, float, dict]] = []
    seen_text: set[str] = set()

    def append_row(row):
        _, _, doc, meta = row
        cleaned = doc.strip()
        if not cleaned:
            return
        key = re.sub(r"\s+", " ", cleaned.lower())
        if key in seen_text:
            return
        seen_text.add(key)
        selected.append((cleaned, 0.0, meta))

    for kind, value in references:
        if kind == "page":
            page_re = re.compile(
                rf"(?i)(?:\[OCR content from PDF page\s+{re.escape(value)}\]|(?:^|\b)page\s+{re.escape(value)}\b|---\s*page\s+{re.escape(value)}\b)"
            )
            matches = [
                row for row in rows
                if page_re.search(row[2]) or str((row[3] or {}).get("page", "")) == str(value)
            ]
            for row in matches[:max_chunks_per_reference]:
                append_row(row)
            if matches:
                print(f"[rag] STRUCTURED REFERENCE page {value}: selected {min(len(matches), max_chunks_per_reference)} chunks", file=sys.stderr)
            continue

        anchor_re = heading_patterns.get(kind)
        if anchor_re is None:
            continue

        anchor_idx = None
        value_norm = str(value).upper()

        # First look for the explicit label (e.g. "Point no 3.2.1").
        for i, row in enumerate(rows):
            match = anchor_re.search(row[2])
            if match and str(match.group(1)).upper() == value_norm:
                anchor_idx = i
                break

        # Many academic/technical PDFs omit the word "section" and render
        # hierarchical headings as just "3.2.1 Title".  For a numbered
        # reference, recover that exact heading directly from the chunk text.
        if anchor_idx is None and re.fullmatch(r"\d+(?:\.\d+)*", str(value)):
            bare_re = re.compile(rf"(?im)^\s*{re.escape(str(value))}(?:\.)?\s+.+$")
            for i, row in enumerate(rows):
                if bare_re.search(row[2]):
                    anchor_idx = i
                    break

        # Metadata fallback: a parent section may not have its own heading chunk
        # in an older index, while descendant chunks (e.g. 3.2.1 / 3.2.2) do.
        # A request for the parent should still retrieve those descendants.
        numeric_value = str(value) if re.fullmatch(r"\d+(?:\.\d+)*", str(value)) else ""
        if anchor_idx is None and numeric_value:
            prefix = numeric_value + "."
            meta_rows = [
                row for row in rows
                if str((row[3] or {}).get("section_number", "") or "") == numeric_value
                or str((row[3] or {}).get("section_number", "") or "").startswith(prefix)
            ]
            if meta_rows:
                selected_meta_docs = meta_rows[:max_chunks_per_reference]
                for row in selected_meta_docs:
                    append_row(row)
                print(
                    f"[rag] STRUCTURED REFERENCE {kind} {value}: metadata fallback selected {len(selected_meta_docs)} chunks",
                    file=sys.stderr,
                )
                continue

        if anchor_idx is None:
            print(f"[rag] STRUCTURED REFERENCE {kind} {value}: no anchor found", file=sys.stderr)
            continue

        target_parts = tuple(int(part) for part in str(value).split(".")) if re.fullmatch(r"\d+(?:\.\d+)*", str(value)) else ()

        end_idx = len(rows)
        for i in range(anchor_idx + 1, len(rows)):
            next_doc = rows[i][2]
            match = anchor_re.search(next_doc)
            if match:
                next_value = str(match.group(1))
                if next_value.upper() != value_norm:
                    end_idx = i
                    break
            elif target_parts:
                # Respect hierarchical boundaries for bare numeric headings:
                # include descendants, stop at the next sibling/ancestor heading.
                m_num = re.search(r"(?im)^\s*(\d+(?:\.\d+)*)(?:\.)?\s+.+$", next_doc)
                if m_num:
                    next_parts = tuple(int(part) for part in m_num.group(1).split("."))
                    if len(next_parts) <= len(target_parts) and next_parts != target_parts:
                        end_idx = i
                        break
                    if len(next_parts) == len(target_parts) and next_parts != target_parts:
                        end_idx = i
                        break


        section_rows = rows[anchor_idx:end_idx]

        # When the index carries section metadata, it is safer to select all
        # chunks belonging to the referenced hierarchical section directly.
        # A parent like 3.2 includes descendants (3.2.1, 3.2.2, ...), while
        # 3.2.1 selects only that exact section.
        numeric_value = str(value) if re.fullmatch(r"\d+(?:\.\d+)*", str(value)) else ""
        if numeric_value:
            selected_by_meta = []
            parent_prefix = numeric_value + "."
            for row in rows:
                meta = row[3] or {}
                meta_num = str(meta.get("section_number", "") or "")
                if meta_num == numeric_value or meta_num.startswith(parent_prefix):
                    selected_by_meta.append(row)
            if selected_by_meta:
                section_rows = selected_by_meta

        for row in section_rows[:max_chunks_per_reference]:
            append_row(row)

        print(
            f"[rag] STRUCTURED REFERENCE {kind} {value}: selected {min(len(section_rows), max_chunks_per_reference)} contiguous chunks",
            file=sys.stderr,
        )

    return ([x[0] for x in selected], [x[1] for x in selected], [x[2] for x in selected])


def _retrieve_summary_chunks(
    collection,
    target_source: Optional[str] = None,
    max_chunks: int = 12,
) -> tuple[list[str], list[float], list[dict]]:
    """Retrieve representative chunks distributed across the document for summary generation.

    Does not depend on keyword matching or the word 'summary'.
    Samples document structure:
      - intro / beginning chunks
      - distributed chunks across middle sections
      - conclusion / final chunks
    """
    try:
        data = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"[rag] Summary retrieval failed: {exc}", file=sys.stderr)
        return [], [], []

    raw_docs = list(data.get("documents") or [])
    raw_metas = list(data.get("metadatas") or [])
    raw_ids = list(data.get("ids") or [])

    if not raw_docs:
        return [], [], []

    filtered = []
    for idx, doc in enumerate(raw_docs):
        meta = raw_metas[idx] if idx < len(raw_metas) else {}
        if target_source and (meta or {}).get("source") != target_source:
            continue
        chunk_id = str(raw_ids[idx]) if idx < len(raw_ids) else str(idx)
        filtered.append((idx, chunk_id, doc or "", meta or {}))

    if not filtered:
        for idx, doc in enumerate(raw_docs):
            meta = raw_metas[idx] if idx < len(raw_metas) else {}
            chunk_id = str(raw_ids[idx]) if idx < len(raw_ids) else str(idx)
            filtered.append((idx, chunk_id, doc or "", meta or {}))

    def _order_key(row):
        original_idx, chunk_id, _, _ = row
        match = re.search(r"_(\d+)$", chunk_id)
        if match:
            return (0, int(match.group(1)))
        return (1, original_idx)

    filtered.sort(key=_order_key)

    total_chunks = len(filtered)
    if total_chunks <= max_chunks:
        selected_rows = filtered
    else:
        selected_indices = {0, 1, total_chunks - 1}
        step = max(1, total_chunks // (max_chunks - 2))
        for i in range(step, total_chunks - 1, step):
            selected_indices.add(i)
        selected_rows = [filtered[i] for i in sorted(selected_indices) if 0 <= i < total_chunks][:max_chunks]

    docs = [row[2] for row in selected_rows if row[2].strip()]
    metas = [row[3] for row in selected_rows if row[2].strip()]
    distances = [0.0] * len(docs)

    dedup_docs, dedup_dists, dedup_metas = _deduplicate_chunks(docs, distances, metas)
    print(
        f"[rag] SUMMARY RETRIEVAL: selected {len(dedup_docs)} representative chunks "
        f"across {total_chunks} total chunks",
        file=sys.stderr,
    )
    return dedup_docs, dedup_dists, dedup_metas


def _retrieve_relevant_chunks(
    collection,
    question: str,
    doc_count: int,
    primary_source: str,
    target_source: Optional[str] = None,
    out_stats: Optional[dict] = None,
) -> tuple[list[str], list[float], list[dict]]:
    """
    Retrieve semantically relevant, deduplicated, and keyword-reranked
    chunks for *question* from *collection*.
    """
    # Generic structured references get section-aware retrieval before vector
    # search. This is not tied to Q1/Q3; it handles natural references to document
    # structure across arbitrary uploaded files.
    structured_refs = _extract_structured_references(question)
    is_image_query = _is_image_content_query(question)
    if structured_refs and not is_image_query:
        section_docs, section_dists, section_metas = _retrieve_structured_sections(
            collection, structured_refs, target_source=target_source, max_chunks_per_reference=80
        )
        if section_docs:
            if out_stats is not None:
                out_stats["candidate_count"] = len(section_docs)
                out_stats["filtered_count"] = len(section_docs)
                out_stats["dedup_count"] = len(section_docs)
                out_stats["reranked_count"] = len(section_docs)
                out_stats["final_chunk_count"] = len(section_docs)
                out_stats["distances"] = [0.0] * len(section_docs)
                out_stats["sources"] = _safe_source_list(section_metas)
                out_stats["structured_references"] = structured_refs
            return section_docs, section_dists, section_metas
    is_plural_image_query = _is_plural_image_content_query(question)

    effective_target_source = target_source
    if is_image_query and not is_plural_image_query and not target_source:
        collection_chat_id = getattr(collection, "name", "") or ""
        effective_target_source = _latest_image_source(
            collection, collection_chat_id
        )

    n_results = min(TOP_K, doc_count)

    t_emb_start = time.perf_counter()
    print("[PERF] embedding start", file=sys.stderr)
    question_embedding = embed_texts([question])[0]
    t_emb_end = time.perf_counter()
    print(f"[PERF] embedding end (duration={t_emb_end - t_emb_start:.3f}s)", file=sys.stderr)

    query_kwargs: dict = {
        "query_embeddings": [question_embedding],
        "n_results": n_results,
        "include": ["documents", "distances", "metadatas"],
    }
    if effective_target_source:
        query_kwargs["where"] = {"source": effective_target_source}

    t_vec_start = time.perf_counter()
    print("[PERF] vector retrieval start", file=sys.stderr)
    query_result = collection.query(**query_kwargs)
    t_vec_end = time.perf_counter()
    print(f"[PERF] vector retrieval end (duration={t_vec_end - t_vec_start:.3f}s)", file=sys.stderr)

    raw_distances: list[float] = query_result["distances"][0] if query_result["distances"] else []
    raw_docs: list[str] = query_result["documents"][0] if query_result["documents"] else []
    raw_metas: list[dict] = query_result.get("metadatas", [[]])[0] if query_result.get("metadatas") else []

    # ── Literal keyword guarantee pass ───────────────────────────
    question_keywords_for_literal_scan = _extract_question_keywords(question)
    if not is_image_query:
        t_lex_start = time.perf_counter()
        print("[PERF] lexical retrieval start", file=sys.stderr)
        literal_docs, literal_dists, literal_metas = _literal_keyword_matches(
            collection, question_keywords_for_literal_scan,
            already_seen=set(raw_docs),
            question_embedding=question_embedding,
            target_source=effective_target_source,
        )
        t_lex_end = time.perf_counter()
        print(f"[PERF] lexical retrieval end (duration={t_lex_end - t_lex_start:.3f}s)", file=sys.stderr)
        if literal_docs:
            raw_docs = raw_docs + literal_docs
            raw_distances = raw_distances + literal_dists
            raw_metas = raw_metas + literal_metas

    if is_image_query:
        image_pairs = [
            (doc, dist, meta)
            for doc, dist, meta in zip(raw_docs, raw_distances, raw_metas)
            if (meta or {}).get("type") == "image" or is_image_file((meta or {}).get("source", ""))
        ]
        if image_pairs:
            raw_docs = [p[0] for p in image_pairs]
            raw_distances = [p[1] for p in image_pairs]
            raw_metas = [p[2] for p in image_pairs]

    # ── Detailed retrieval log ──────────────────────────────────
    print(
        f"\n[rag] ══════════════════════════════════════════",
        file=sys.stderr,
    )
    print(f"[rag] QUESTION: {question}", file=sys.stderr)
    if is_image_query:
        print(
            f"[rag] IMAGE QUERY: plural={is_plural_image_query} "
            f"target_source={effective_target_source or 'all images'}",
            file=sys.stderr,
        )
    print(
        f"[rag] Retrieved {len(raw_docs)} candidate chunks from ChromaDB",
        file=sys.stderr,
    )
    for i, (doc, dist) in enumerate(zip(raw_docs, raw_distances)):
        print(
            f"[rag]   [{i+1}] dist={dist:.4f} | {doc[:100].replace(chr(10), ' ')}",
            file=sys.stderr,
        )

    # ── Step 1: relevance filtering (primary pass) ──────────────
    substantive_kws = [
        kw for kw in question_keywords_for_literal_scan
        if len(kw) >= 2 and kw not in {
            "candidate", "person", "information", "document", "tell", "tells",
            "telling", "told", "details", "what", "where", "when", "about",
            "give", "gives", "show", "shows", "state", "states", "describe",
            "describes", "mention", "mentions", "find", "finds",
        }
    ]

    def _apply_threshold(
        base_thresh: float,
        docs: list[str],
        distances: list[float],
        metas: list[dict],
        adaptive: bool = True,
    ) -> tuple[list[str], list[float], list[dict]]:
        """Return items whose distance is within the threshold (adaptive by default)."""
        f_docs, f_dists, f_metas = [], [], []
        for d, dist, m in zip(docs, distances, metas):
            if is_image_query and (m or {}).get("type") == "image":
                cutoff = 1.0
            else:
                cutoff = get_adaptive_threshold(d, base_thresh) if adaptive else base_thresh

            # Hybrid score check: evaluate keyword presence
            ks = _keyword_score(d, substantive_kws) if substantive_kws else 0.0
            hybrid_dist = max(0.0, dist - 0.20 * ks)

            # Pass if raw vector distance <= cutoff, or hybrid distance <= cutoff
            # provided raw vector distance does not exceed RELEVANCE_THRESHOLD_FALLBACK (0.78)
            if dist <= cutoff or (hybrid_dist <= cutoff and dist <= RELEVANCE_THRESHOLD_FALLBACK):
                f_docs.append(d)
                f_dists.append(dist)
                f_metas.append(m)
            else:
                print(
                    f"[rag] FILTER: removed dist={dist:.4f} (hybrid={hybrid_dist:.4f}) > cutoff={cutoff:.4f} "
                    f"(len={len(d)}) | {d[:60].replace(chr(10), ' ')}",
                    file=sys.stderr,
                )
        return f_docs, f_dists, f_metas

    filtered_docs, filtered_distances, filtered_metas = _apply_threshold(
        RELEVANCE_THRESHOLD, raw_docs, raw_distances, raw_metas, adaptive=True
    )

    # ── Fallback pass: loosen threshold if primary pass found nothing ──
    # This handles vague queries ("tell", "phone number") and documents
    # where all chunks sit slightly above the primary threshold.
    if not filtered_docs:
        print(
            f"[rag] Primary pass empty (best dist={raw_distances[0]:.4f}). "
            f"Trying fallback threshold {RELEVANCE_THRESHOLD_FALLBACK}...",
            file=sys.stderr,
        )
        filtered_docs, filtered_distances, filtered_metas = _apply_threshold(
            RELEVANCE_THRESHOLD_FALLBACK, raw_docs, raw_distances, raw_metas, adaptive=False
        )

    if not filtered_docs:
        print(
            f"[rag] No chunks passed even fallback threshold {RELEVANCE_THRESHOLD_FALLBACK}. "
            f"Best distance was {raw_distances[0]:.4f}.",
            file=sys.stderr,
        )
        if out_stats is not None:
            out_stats["candidate_count"] = len(raw_docs)
            out_stats["filtered_count"] = 0
            out_stats["dedup_count"] = 0
            out_stats["reranked_count"] = 0
            out_stats["final_chunk_count"] = 0
            out_stats["distances"] = []
            out_stats["sources"] = []
        return [], [], []

    # ── Step 2: deduplication ───────────────────────────────────
    deduped_docs, deduped_distances, deduped_metas = _deduplicate_chunks(
        filtered_docs, filtered_distances, filtered_metas
    )

    # ── Step 3: keyword reranking ───────────────────────────────
    keywords = _extract_question_keywords(question)
    print(
        f"[rag] Question keywords (expanded): {keywords}",
        file=sys.stderr,
    )

    final_docs, final_distances, final_metas = _rerank_chunks(
        deduped_docs, deduped_distances, deduped_metas,
        question, top_n=RERANK_TOP_N,
    )

    if out_stats is not None:
        out_stats["candidate_count"] = len(raw_docs)
        out_stats["filtered_count"] = len(filtered_docs)
        out_stats["dedup_count"] = len(deduped_docs)
        out_stats["reranked_count"] = len(final_docs)
        out_stats["final_chunk_count"] = len(final_docs)
        out_stats["distances"] = [round(float(d), 4) for d in final_distances]
        out_stats["sources"] = _safe_source_list(final_metas)

    print(
        f"[rag] FINAL: {len(final_docs)} chunks "
        f"(candidates={len(raw_docs)} -> filtered={len(filtered_docs)} -> "
        f"deduped={len(deduped_docs)} -> reranked_top={len(final_docs)})",
        file=sys.stderr,
    )
    print(
        f"[rag] ══════════════════════════════════════════\n",
        file=sys.stderr,
    )

    return final_docs, final_distances, final_metas

def _safe_truncate_text(text: str, max_chars: int) -> str:
    """Safely truncate text at paragraph, newline, or sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.6:
        return truncated[:last_para].strip()
    last_line = truncated.rfind("\n")
    if last_line > max_chars * 0.6:
        return truncated[:last_line].strip()
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1].strip()
    return truncated.strip()

def _build_bounded_context(
    docs: list[str],
    metas: list[dict],
    primary_source: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict]]:
    """
    Assemble evidence chunks into a clean, bounded context string.

    Guarantees:
    - Never exceeds max_chars.
    - Truncates safely at CHUNK BOUNDARIES: if adding the next chunk would
      exceed max_chars, it is skipped cleanly rather than cut in half mid-evidence.
    - If a single first chunk exceeds max_chars, it is safely truncated at a
      clean line/sentence boundary instead of discarding it.
    - Returns formatted context string and deduplicated sources list.
    """
    t_ctx_start = time.perf_counter()
    print("[PERF] context construction start", file=sys.stderr)
    context_parts: list[str] = []
    source_chunks: list[tuple[str, str]] = []
    current_chars = 0

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        cleaned_doc = doc.strip()
        if not cleaned_doc:
            continue
        src = (meta or {}).get("source", primary_source) or primary_source or "document"
        entry = f"[Evidence {i} (Source: {src})]\n{cleaned_doc}"
        sep_len = len("\n\n---\n\n") if context_parts else 0
        candidate_len = current_chars + sep_len + len(entry)

        if candidate_len <= max_chars:
            context_parts.append(entry)
            source_chunks.append((src, cleaned_doc[:300]))
            current_chars = candidate_len
        else:
            if not context_parts:
                allowed = max(200, max_chars - len(f"[Evidence 1 (Source: {src})]\n"))
                truncated_text = _safe_truncate_text(cleaned_doc, allowed)
                entry = f"[Evidence 1 (Source: {src})]\n{truncated_text}"
                context_parts.append(entry)
                source_chunks.append((src, truncated_text[:300]))
                print(
                    f"[rag] First chunk exceeded max_chars ({len(cleaned_doc)} > {max_chars}). "
                    f"Truncated safely at boundary ({len(truncated_text)} chars).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[rag] Context limit reached ({current_chars}/{max_chars} chars). "
                    f"Truncating at chunk boundary; omitting chunk {i} ({len(cleaned_doc)} chars).",
                    file=sys.stderr,
                )
            break

    context = "\n\n---\n\n".join(context_parts)
    seen: set[str] = set()
    sources_out: list[dict] = []
    for label, snippet in source_chunks:
        if label not in seen:
            seen.add(label)
            sources_out.append({"label": label, "snippet": snippet})

    t_ctx_end = time.perf_counter()
    print(f"[PERF] context construction end (duration={t_ctx_end - t_ctx_start:.3f}s)", file=sys.stderr)
    return context, sources_out

def _clean_chunk_text(text: str) -> str:
    """Strip any legacy control strings or headers that may have been indexed."""
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"\[Evidence\s+\d+\s*\(Source:.*?\)\]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[IMAGE OCR CONTENT[^\n]*\]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[FILE:\s*[^\]]+?\s*\((?:Image|Document)\)\]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"QUERY RETRIEVAL SCOPE[^\n]*\n?", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def _build_image_context(
    docs: list[str],
    metas: list[dict],
    target_source: Optional[str] = None,
    image_sources: Optional[list[str]] = None,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict]]:
    """Assemble OCR content cleanly for querymind-image-qa without internal control strings.

    Guarantees:
    - Strips all legacy [FILE: ...], [Evidence ...], [IMAGE OCR CONTENT ...] headers.
    - If a single image is present/targeted, produces clean OCR text without metadata clutter.
    - If multiple images are present, uses minimal clean headers (e.g. 'Image 1 (filename):')
      to distinguish them clearly for the LLM.
    - Bounded safely to max_chars.
    """
    clean_items: list[tuple[str, str]] = []
    seen_texts: set[str] = set()

    for doc, meta in zip(docs, metas):
        src = (meta or {}).get("source", "")
        cleaned = _clean_chunk_text(doc)
        if not cleaned:
            continue
        norm_key = re.sub(r"\s+", " ", cleaned.lower())
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)
        clean_items.append((src, cleaned))

    if not clean_items:
        return "", []

    unique_sources = list(dict.fromkeys(src for src, _ in clean_items if src))

    context_parts: list[str] = []
    sources_out: list[dict] = []
    current_chars = 0

    is_multi_image_chat = bool(image_sources and len(image_sources) > 1)

    if is_multi_image_chat or len(unique_sources) > 1:
        # Multiple images in chat or retrieval: group chunks by image with minimal clean labeling
        grouped: dict[str, list[str]] = {}
        for src, text in clean_items:
            grouped.setdefault(src, []).append(text)

        img_list = image_sources or unique_sources
        for src, chunk_list in grouped.items():
            img_num = (img_list.index(src) + 1) if (img_list and src in img_list) else ""
            num_label = f"Image {img_num} ({src})" if img_num else f"Image ({src})"
            combined_chunks = "\n\n".join(chunk_list)
            entry = f"{num_label}:\n{combined_chunks}"
            sep = "\n\n---\n\n" if context_parts else ""
            if current_chars + len(sep) + len(entry) <= max_chars:
                context_parts.append(entry)
                current_chars += len(sep) + len(entry)
                sources_out.append({"label": src, "snippet": combined_chunks[:300]})
            else:
                if not context_parts:
                    truncated = _safe_truncate_text(entry, max_chars)
                    context_parts.append(truncated)
                    sources_out.append({"label": src, "snippet": truncated[:300]})
                break
    else:
        # Single image in chat: join chunks cleanly without internal control headers
        for src, text in clean_items:
            sep = "\n\n" if context_parts else ""
            if current_chars + len(sep) + len(text) <= max_chars:
                context_parts.append(text)
                current_chars += len(sep) + len(text)
                if src and not any(s["label"] == src for s in sources_out):
                    sources_out.append({"label": src, "snippet": text[:300]})
            else:
                if not context_parts:
                    truncated = _safe_truncate_text(text, max_chars)
                    context_parts.append(truncated)
                    if src:
                        sources_out.append({"label": src, "snippet": truncated[:300]})
                break

    return "\n\n".join(context_parts), sources_out

def _retrieve_image_chunks(
    collection,
    target_source: Optional[str] = None,
    image_sources: Optional[list[str]] = None,
    question: Optional[str] = None,
) -> tuple[list[str], list[float], list[dict]]:
    """Retrieve chunks for image QA.

    If a specific target_source is given, all indexed chunks for that image
    are returned in natural order so full OCR text is preserved without
    arbitrary distance cutoffs.
    """
    try:
        if target_source:
            res = collection.get(
                where={"source": target_source},
                include=["documents", "metadatas"],
            )
            docs = res.get("documents", []) or []
            metas = res.get("metadatas", []) or []
            if docs:
                distances = [0.0] * len(docs)
                print(f"[retrieval] image_chunks={len(docs)} (target='{target_source}')", file=sys.stderr)
                return docs, distances, metas

        # If no single target, gather from all available image_sources
        query_where = {}
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs = all_data.get("documents", []) or []
        all_metas = all_data.get("metadatas", []) or []

        matched_docs = []
        matched_metas = []
        for d, m in zip(all_docs, all_metas or [{}] * len(all_docs)):
            m = m or {}
            src = m.get("source", "")
            if m.get("type") == "image" or is_image_file(src):
                if not image_sources or src in image_sources:
                    matched_docs.append(d)
                    matched_metas.append(m)

        distances = [0.0] * len(matched_docs)
        print(f"[retrieval] image_chunks={len(matched_docs)} across all images", file=sys.stderr)
        return matched_docs, distances, matched_metas

    except Exception as exc:
        print(f"[retrieval] image chunk fetch error: {exc}", file=sys.stderr)
        return [], [], []

def _detect_target_source(
    question: str,
    available_sources: list[str],
    explicit_source: Optional[str] = None,
    image_sources: Optional[list[str]] = None,
    document_sources: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Identify whether a question targets a specific document or image in a multi-source chat.

    Resolution order:
    1. If explicit_source is passed and matches an available source, use it.
    2. Positional/ordinal resolution (e.g. 'first image', '2nd screenshot', 'second pdf').
    3. Exact filename or stem match in question.
    4. Distinctive stem word match in question.
    5. Returns matching source filename, or None if the query is general.
    """
    if not available_sources:
        return None

    # 1. Explicit source
    if explicit_source:
        norm_exp = os.path.basename(explicit_source.strip()).lower()
        for src in available_sources:
            norm_src = os.path.basename(str(src).strip()).lower()
            if norm_src == norm_exp:
                return src

    norm_q = question.lower().strip()

    # 2. Positional / Ordinal resolution
    ordinal_map = {
        "first": 0, "1st": 0, "one": 0, "1": 0,
        "second": 1, "2nd": 1, "two": 1, "2": 1,
        "third": 2, "3rd": 2, "three": 2, "3": 2,
        "fourth": 3, "4th": 3, "four": 3, "4": 3,
        "fifth": 4, "5th": 4, "five": 4, "5": 4,
        "last": -1, "latest": -1, "recent": -1,
    }

    # Positional check for images
    if image_sources:
        img_ord_m = re.search(
            r"\b(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|latest)\s+"
            r"(?:image|picture|photo|screenshot|graphic|illustration|figure|pic)\b",
            norm_q,
        )
        if not img_ord_m:
            img_ord_m = re.search(
                r"\b(?:image|picture|photo|screenshot|pic)\s*(?:#|number\s+)?([1-5])\b",
                norm_q,
            )
        if img_ord_m:
            ord_word = img_ord_m.group(1).lower()
            if ord_word in ordinal_map:
                idx = ordinal_map[ord_word]
                if idx == -1 and image_sources:
                    return image_sources[-1]
                if 0 <= idx < len(image_sources):
                    return image_sources[idx]

    # Positional check for documents
    if document_sources:
        doc_ord_m = re.search(
            r"\b(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|latest)\s+"
            r"(?:document|doc|pdf|file|paper|report)\b",
            norm_q,
        )
        if not doc_ord_m:
            doc_ord_m = re.search(
                r"\b(?:document|doc|pdf|file)\s*(?:#|number\s+)?([1-5])\b",
                norm_q,
            )
        if doc_ord_m:
            ord_word = doc_ord_m.group(1).lower()
            if ord_word in ordinal_map:
                idx = ordinal_map[ord_word]
                if idx == -1 and document_sources:
                    return document_sources[-1]
                if 0 <= idx < len(document_sources):
                    return document_sources[idx]

    # 3. Exact source filename match in question
    # The user must explicitly mention the exact filename (e.g. 'resume.pdf' or 'sales_report.docx')
    for src in available_sources:
        base = os.path.basename(src).lower()
        pattern = r"(?:^|[\s\"'‘“(\[{<])" + re.escape(base) + r"(?:$|[\s\"'’”\]}>?,.!;:?])"
        if re.search(pattern, norm_q) or base in norm_q:
            return src

    # 4. Safe natural-language filename alias match.
    # Users commonly say "resume", "my resume", "presentation", etc.
    # even when the stored filename is something like "Muhammad Waleed Resume.pdf".
    # Only resolve an alias when it uniquely identifies one available source; this
    # prevents a generic word from accidentally narrowing a multi-document query.
    generic = {
        "file", "files", "document", "documents", "pdf", "pdfs",
        "doc", "docx", "ppt", "pptx", "report", "paper", "check",
        "please", "the", "this", "that", "in", "on", "from", "for",
    }
    q_tokens = [
        tok for tok in re.findall(r"[a-z0-9]+", norm_q)
        if len(tok) >= 4 and tok not in generic
    ]
    alias_matches: list[str] = []
    for src in available_sources:
        stem = os.path.splitext(os.path.basename(str(src).strip()).lower())[0]
        stem_tokens = set(re.findall(r"[a-z0-9]+", stem))
        if not stem_tokens:
            continue
        # A token match is accepted only for a meaningful source-name token.
        # Example: query "resume.pdf file check" -> unique token "resume".
        if any(tok in stem_tokens for tok in q_tokens):
            alias_matches.append(src)

    if len(alias_matches) == 1:
        print(
            f"[rag] Natural filename alias selected unique source: '{alias_matches[0]}'",
            file=sys.stderr,
        )
        return alias_matches[0]

    # Otherwise, return None to search across ALL documents in the chat
    return None


def _find_document_on_disk(source_name: str, chat_id: str) -> Optional[str]:
    """Locate the document or image file on disk for a given chat_id and source name."""
    if not source_name:
        return None
    safe_name = os.path.basename(source_name)

    # 1. Check chat storage directory directly
    chat_path = STORAGE_DIR / chat_id / safe_name
    if chat_path.is_file():
        return str(chat_path)

    # 2. Check user documents
    for match in (STORAGE_DIR / "user_documents").glob(f"*/{safe_name}"):
        if match.is_file():
            return str(match)

    # 3. Check any directory in storage
    for match in STORAGE_DIR.glob(f"*/{safe_name}"):
        if match.is_file():
            return str(match)

    # 4. If source_name is already a path that exists
    if os.path.isfile(source_name):
        return os.path.abspath(source_name)

    return None


def _extract_figure_reference(question: str) -> Optional[int]:
    """Return an explicitly requested figure number, when present.

    Handles natural variants such as ``Figure 5``, ``Fig. 5``, ``figure no. 5``
    and ``figure number 5`` without relying on any document-specific numbering.
    """
    q = re.sub(r"\s+", " ", (question or "").strip())
    patterns = (
        r"\bfig(?:ure)?\.?\s*(?:no\.?|number)?\s*(\d+)\b",
        r"\b(?:image|diagram|chart|graph|illustration)\s+(?:no\.?|number)?\s*(\d+)\b",
    )
    for pattern in patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _figure_pages_from_pdf(file_path: str, figure_number: int) -> list[int]:
    """Find pages containing the requested figure caption.

    Results are ordered so pages with actual visual content (embedded images or
    vector drawings) are preferred over continuation pages that merely mention
    the figure again in running text.
    """
    if not file_path or not os.path.isfile(file_path):
        return []
    if os.path.splitext(file_path)[1].lower() != ".pdf":
        return []

    try:
        import pymupdf  # type: ignore
        doc = pymupdf.open(file_path)
        try:
            caption_re = re.compile(
                rf"\bfigure\s*{figure_number}\s*[:.\-–—]?", re.IGNORECASE
            )
            fig_re = re.compile(rf"\bfig\.?\s*{figure_number}\b", re.IGNORECASE)
            scored: list[tuple[int, int, int]] = []
            for page_idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if not (caption_re.search(text) or fig_re.search(text)):
                    continue
                visual_score = min(len(page.get_images(full=True)), 5) * 10
                try:
                    visual_score += min(len(page.get_drawings()), 10)
                except Exception:
                    pass
                caption_count = len(caption_re.findall(text)) + len(fig_re.findall(text))
                # Prefer real visual pages, then more explicit caption occurrences.
                scored.append((page_idx, visual_score, caption_count))
            scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
            return [row[0] for row in scored]
        finally:
            doc.close()
    except Exception as exc:
        print(f"[retrieval] figure-caption scan failed: {exc}", file=sys.stderr)
        return []


def _all_figure_caption_pages_from_pdf(file_path: str) -> list[int]:
    """Return pages that contain a numbered Figure/Fig. caption, ranked visually."""
    if not file_path or not os.path.isfile(file_path):
        return []
    if os.path.splitext(file_path)[1].lower() != ".pdf":
        return []

    try:
        import pymupdf  # type: ignore
        doc = pymupdf.open(file_path)
        try:
            caption_re = re.compile(r"\b(?:figure|fig)\.?\s*\d+\s*[:.\-–—]", re.IGNORECASE)
            scored: list[tuple[int, int, int]] = []
            for page_idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                matches = caption_re.findall(text)
                if not matches:
                    continue
                visual_score = min(len(page.get_images(full=True)), 5) * 10
                try:
                    visual_score += min(len(page.get_drawings()), 10)
                except Exception:
                    pass
                scored.append((page_idx, visual_score, len(matches)))
            scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
            return [row[0] for row in scored]
        finally:
            doc.close()
    except Exception as exc:
        print(f"[retrieval] all-figure caption scan failed: {exc}", file=sys.stderr)
        return []


def _figure_pages_for_question(
    question: str,
    chat_id: str,
    target_source: Optional[str],
    available_sources: list[str],
) -> list[tuple[str, int]]:
    """Resolve visual figure references directly to document page(s)."""
    figure_number = _extract_figure_reference(question)
    candidate_sources = [target_source] if target_source else list(available_sources)
    resolved: list[tuple[str, int]] = []

    for src in candidate_sources:
        if not src:
            continue
        path = _find_document_on_disk(src, chat_id)
        if not path:
            continue
        if figure_number is not None:
            pages = _figure_pages_from_pdf(path, figure_number)
        else:
            pages = _all_figure_caption_pages_from_pdf(path)
        resolved.extend((src, page_num) for page_num in pages)

    return resolved


def _locate_candidate_pages(
    collection,
    question: str,
    chat_id: str,
    target_source: Optional[str] = None,
    max_pages: int = 2,
) -> list[dict]:
    """
    Locate candidate pages (with preserved page images) for generic multimodal visual QA.

    Uses:
    1. Explicit structured page references in the question (e.g. 'page 7', 'page 3').
    2. ChromaDB vector retrieval across chunks to identify relevant page numbers from chunk metadata.
    3. Resolves each page to its preserved page image on disk.
    """
    t_start = time.perf_counter()
    print(f"[PERF] page retrieval start - question='{question[:60]}...' chat_id={chat_id}", file=sys.stderr)

    candidate_pages: list[tuple[str, int]] = []

    # 0. Explicit figure/diagram/chart references take priority over semantic
    # retrieval. A query such as "Explain Figure 5" must resolve to the page
    # containing Figure 5, not the page that happens to score highest for the
    # generic word "figure".
    try:
        all_meta = collection.get(include=["metadatas"])
        available_sources = []
        seen_sources: set[str] = set()
        for meta in all_meta.get("metadatas", []) or []:
            if not isinstance(meta, dict):
                continue
            src = str(meta.get("source", "")).strip()
            if src and src not in seen_sources:
                seen_sources.add(src)
                available_sources.append(src)
        figure_pages = _figure_pages_for_question(
            question,
            chat_id,
            target_source,
            available_sources,
        )
        if figure_pages:
            candidate_pages.extend(figure_pages)
            print(
                f"[retrieval] FIGURE REFERENCE RESOLVED: question={question!r} -> {figure_pages}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[retrieval] figure-reference resolution failed: {exc}", file=sys.stderr)

    # 1. Check explicit page references
    refs = _extract_structured_references(question)
    ref_pages = [val for kind, val in refs if kind == "page"]
    for ref_p in ref_pages:
        try:
            page_num = int(ref_p)
            if target_source:
                candidate_pages.append((target_source, page_num))
            else:
                all_chunks = collection.get(include=["metadatas"])
                sources = set(m.get("source") for m in (all_chunks.get("metadatas") or []) if m.get("source"))
                for s in sources:
                    candidate_pages.append((s, page_num))
        except (ValueError, TypeError):
            pass

    # 2. ChromaDB retrieval to locate candidate chunks and their page numbers
    # Skip semantic page discovery when an explicit figure was already resolved.
    # Otherwise the word "figure" can pull the wrong page and defeat visual QA.
    where_clause = None
    if target_source:
        where_clause = {"source": target_source}

    try:
        if candidate_pages and _extract_figure_reference(question) is not None:
            raise RuntimeError("skip semantic page discovery after explicit figure resolution")
        q_emb = embed_texts([question])[0]
        count = collection.count()
        if count > 0:
            query_kwargs = {
                "query_embeddings": [q_emb],
                "n_results": min(TOP_K, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_clause:
                query_kwargs["where"] = where_clause

            results = collection.query(**query_kwargs)
            if results and results.get("metadatas") and results["metadatas"][0]:
                metas = results["metadatas"][0]
                docs = results["documents"][0]
                for meta, doc in zip(metas, docs):
                    src = meta.get("source")
                    pg = meta.get("page")
                    if not pg:
                        m = re.search(r"\[OCR content from PDF page\s+(\d+)\]", doc)
                        if m:
                            pg = int(m.group(1))
                        else:
                            pg = 1
                    try:
                        pg = int(pg)
                    except (ValueError, TypeError):
                        pg = 1
                    if src and (src, pg) not in candidate_pages:
                        candidate_pages.append((src, pg))
    except Exception as exc:
        print(f"[retrieval] Candidate page query failed: {exc}", file=sys.stderr)

    # Fallback if no candidate pages found yet
    if not candidate_pages:
        try:
            peek_data = collection.get(limit=10, include=["metadatas"])
            for meta in peek_data.get("metadatas", []):
                src = meta.get("source")
                pg = int(meta.get("page", 1))
                if src and (src, pg) not in candidate_pages:
                    candidate_pages.append((src, pg))
                    break
        except Exception:
            pass

    # Resolve each candidate page to an image path on disk
    resolved_candidates: list[dict] = []
    primary_src = target_source or (candidate_pages[0][0] if candidate_pages else None)

    for src, pg in candidate_pages:
        if len(resolved_candidates) >= max_pages:
            break
        # If we already have a candidate for the primary document, do not mix other unrelated documents
        if primary_src and src != primary_src and len(resolved_candidates) > 0:
            continue
        doc_path = _find_document_on_disk(src, chat_id)
        if not doc_path:
            continue
        img_path = get_or_render_page_image(doc_path, pg)
        if img_path and os.path.isfile(img_path):
            resolved_candidates.append({
                "source": src,
                "page": pg,
                "image_path": img_path,
                "doc_path": doc_path,
            })

    t_end = time.perf_counter()
    print(
        f"[PERF] page retrieval end (duration={t_end - t_start:.3f}s) - "
        f"resolved {len(resolved_candidates)} page(s): {[(c['source'], c['page']) for c in resolved_candidates]}",
        file=sys.stderr,
    )
    return resolved_candidates

