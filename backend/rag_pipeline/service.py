from __future__ import annotations
import contextlib
import os
import re
import sys
import time
from typing import Optional

from .common import MAX_CONTEXT_CHARS, MAX_TOKENS, RERANK_TOP_N, _NO_DOCUMENT_MSG, _groq_client, _not_covered_msg, GROQ_API_KEY, _BACKEND_DIR
from .document_structure import extract_headings_from_source_text, extract_numbered_section, _iter_numbered_heading_candidates, extract_reference_entries
from backend.parsers.document_parser import extract_text as _extract_source_text
from .document_structure import extract_numbered_headings_from_pdf_file

from .query import (
    _extract_definition_term,
    _extract_topic_term,
    _extract_informational_topic,
    _is_informational_query,
    _is_heading_extraction_query,
    _is_specific_heading_lookup_query,
    _parse_section_target,
    _extract_target_section,
    _extract_document_headings,
    _determine_query_mode,
    _find_term_in_entire_document,
    _find_topic_in_entire_document,
    _safe_source_list,
    _normalise_for_term_search,
    _is_visual_intent,
    _extract_structured_references,
    _is_reference_count_query,
    _is_reference_list_query,
    _resolve_continuation_question,
    classify_query_intent,
    is_image_file,
)
from .retrieval import (
    _build_bounded_context,
    _build_image_context,
    _detect_target_source,
    _extract_question_keywords,
    _retrieve_relevant_chunks,
    _retrieve_image_chunks,
    _retrieve_summary_chunks,
    _locate_candidate_pages,
    _retrieve_named_section,
)
from .generation import (
    _langfuse_track_generation,
    generate_general_chat_response,
    generate_visual_qa_response,
)
from .intent_router import classify_intent
from .tracing import _anonymize_user, _get_managed_prompt, _langfuse_enabled, _langfuse_track_retrieval, _langfuse, propagate_attributes
from .vector_store import get_chroma_collection

def _is_broad_document_request(question: str) -> bool:
    q = re.sub(r"\s+", " ", (question or "").casefold())
    return bool(re.search(r"\b(?:all|every|full|complete|entire|whole|details?|deep|explain|why|how|compare|remaining|rest|more)\b", q))


def _is_full_section_request(question: str) -> bool:
    q = re.sub(r"\s+", " ", (question or "").casefold())
    return bool(re.search(
        r"\b(?:full|complete|entire|whole|all|everything|remaining|rest|from the file|from the document|"
        r"only|give me|tell me about|tell me|read|show me)\b", q
    ))

def _normalise_section_label(value: str) -> str:
    """Normalize section labels for exact, document-derived matching."""
    return re.sub(r"[^a-z0-9]+", " ", (value or "").casefold()).strip()


def _section_index_from_collection(
    docs: list[str], metas: list[dict], target_source: Optional[str] = None
) -> tuple[set[str], set[str]]:
    """Build a document-derived section index (numbers, normalized titles).

    This uses stored section metadata first and falls back to strong heading lines
    in chunk text. It deliberately does not contain a hardcoded list of names.
    """
    numbers: set[str] = set()
    titles: set[str] = set()
    numbered_re = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.)?\s+(.+?)\s*$")
    structural_re = re.compile(
        r"^\s*(?:Section|Chapter|Part|Module|Appendix)\s+"
        r"(\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])\s*[:.\-–—]?\s*(.*)$",
        re.IGNORECASE,
    )
    section_prefix_re = re.compile(r"^Section:\s*(.+?)\s*$", re.IGNORECASE)

    for idx, doc in enumerate(docs):
        meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        src = str(meta.get("source", ""))
        if target_source and os.path.basename(src).casefold() != os.path.basename(target_source).casefold():
            continue

        sec_num = str(meta.get("section_number", "") or "").strip()
        sec_title = str(meta.get("section_title", "") or "").strip()
        if sec_num:
            numbers.add(sec_num.casefold())
        if sec_title:
            titles.add(_normalise_section_label(sec_title))

        lines = [line.strip() for line in str(doc or "").splitlines() if line.strip()]
        if not lines:
            continue
        first = lines[0]
        m = numbered_re.match(first)
        if m:
            numbers.add(m.group(1).casefold())
            titles.add(_normalise_section_label(m.group(2)))
        m = structural_re.match(first)
        if m:
            numbers.add(m.group(1).casefold())
            if m.group(2).strip():
                titles.add(_normalise_section_label(m.group(2)))
        m = section_prefix_re.match(first)
        if m:
            label = m.group(1).strip()
            # Do not index generated legacy PROFILE/REFERENCE wrappers as real
            # section titles. Other document-derived labels are valid.
            if label.casefold() not in {"profile", "reference"}:
                titles.add(_normalise_section_label(label))

    return numbers, {t for t in titles if t}


def _generic_section_target_is_known(
    parsed: Optional[dict], docs: list[str], metas: list[dict], target_source: Optional[str] = None
) -> bool:
    """Validate a title/number target against the actual uploaded document.

    This is the key guard against false section routing for arbitrary entities such
    as people ("Give me Lukasz Kaiser").
    """
    if not parsed:
        return False
    numbers, titles = _section_index_from_collection(docs, metas, target_source=target_source)
    if parsed.get("number"):
        return str(parsed["number"]).casefold() in numbers
    title = _normalise_section_label(str(parsed.get("title") or ""))
    if not title:
        return False
    if title in titles:
        return True
    # Conservative token containment, but only after both sides are known
    # document headings. This avoids fuzzy-matching arbitrary people/entities.
    title_tokens = set(title.split())
    for known in titles:
        known_tokens = set(known.split())
        if title_tokens and known_tokens and (title_tokens <= known_tokens or known_tokens <= title_tokens):
            return True
    return False


def _answer_question_impl(
    chat_id: str,
    question: str,
    selected_source: Optional[str] = None,
    user_id: Optional[str] = None,
    root_obs=None,
    conversation_history: Optional[list[dict]] = None,
    user_email: Optional[str] = None,
) -> dict:
    t_req_start = time.perf_counter()
    print("[PERF] request start", file=sys.stderr)
    try:
        return _answer_question_core(
            chat_id=chat_id,
            question=question,
            selected_source=selected_source,
            user_id=user_id,
            root_obs=root_obs,
            conversation_history=conversation_history,
            user_email=user_email,
        )
    finally:
        t_req_end = time.perf_counter()
        print(f"[PERF] request total (duration={t_req_end - t_req_start:.3f}s)", file=sys.stderr)

def _load_original_source_text(
    chat_id: str,
    source_name: str,
    user_email: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Load the original source text reliably from local chat storage or Supabase.

    Structured document features must not depend on the local filesystem because
    the original file is also persisted in Supabase and can outlive a backend
    restart/replacement. Returns (text, local_temp_path).
    """
    safe_name = os.path.basename(source_name or "").strip()
    if not safe_name:
        return "", None

    local_path = os.path.join(str(_BACKEND_DIR), "storage", str(chat_id), safe_name)
    if os.path.isfile(local_path) and _extract_source_text is not None:
        try:
            return _extract_source_text(local_path), local_path
        except Exception as exc:
            print(f"[source] local extraction failed for '{safe_name}': {exc}", file=sys.stderr)

    if user_email:
        try:
            import backend.database.storage as _storage
            data, _meta = _storage.download_user_document(user_email, safe_name)
            ext = os.path.splitext(safe_name)[1].lower()
            temp_root = os.path.join(str(_BACKEND_DIR), "storage", str(chat_id), "_source_cache")
            os.makedirs(temp_root, exist_ok=True)
            cache_path = os.path.join(temp_root, safe_name)
            if not os.path.isfile(cache_path) or os.path.getsize(cache_path) != len(data):
                with open(cache_path, "wb") as f:
                    f.write(data)
            if _extract_source_text is not None:
                return _extract_source_text(cache_path), cache_path
        except Exception as exc:
            print(f"[source] Supabase source retrieval failed for '{safe_name}': {exc}", file=sys.stderr)

    return "", None


def _answer_question_core(
    chat_id: str,
    question: str,
    selected_source: Optional[str] = None,
    user_id: Optional[str] = None,
    root_obs=None,
    conversation_history: Optional[list[dict]] = None,
    user_email: Optional[str] = None,
) -> dict:
    """Answer a document question, with controlled definition-mode knowledge access.

    Definition rule:
      1. A clear definition question ("What is X?", "Define X", etc.) is detected.
      2. X is searched across the ENTIRE indexed document, not only retrieved chunks.
      3. If X exists, general knowledge may be used ONLY to define/explain X.
      4. If X does not exist, no outside definition is given.

    Multi-document & source-targeting rules:
      - If a document is explicitly selected or referenced in the question, retrieval
        is restricted to that document.
      - For general questions, the best chunks across all documents are retrieved.
      - Context is strictly bounded to MAX_CONTEXT_CHARS and truncated at chunk boundaries.
    """

    # Conversational follow-ups ("remaining", "tell me more", "the rest") are resolved
    # from recent chat history, but the resolved question still triggers a FRESH document
    # retrieval. This prevents stale/previous chunks from becoming the search scope.
    working_question, previous_user_question, previous_assistant_answer = _resolve_continuation_question(
        question, conversation_history
    )
    if working_question != question:
        print(f"[conversation] follow-up resolved: '{question}' -> '{working_question}'", file=sys.stderr)
    # Route once before any ChromaDB/RAG work. Greetings and casual conversation
    # never touch document retrieval; document questions continue unchanged.
    t_intent_start = time.perf_counter()
    print("[PERF] intent start", file=sys.stderr)
    intent = classify_intent(
        working_question,
        selected_source=selected_source,
        root_obs=root_obs,
    )
    t_intent_end = time.perf_counter()
    print(f"[PERF] intent end (duration={t_intent_end - t_intent_start:.3f}s)", file=sys.stderr)
    if root_obs is not None:
        try:
            root_obs.update(
                metadata={
                    "intent": intent.intent,
                    "intent_confidence": intent.confidence,
                    "mode": "general_chat" if intent.intent != "document_query" else "document_qa",
                }
            )
        except Exception:
            pass

    if intent.intent != "document_query":
        # Python control-flow safeguard: prevent informational queries from bypassing RAG
        safeguard_topic = _extract_informational_topic(working_question)
        if safeguard_topic or _is_informational_query(working_question):
            print(
                f"[router] Python safeguard: overridden '{intent.intent}' to 'document_query' "
                f"for informational question: '{question}'",
                file=sys.stderr,
            )
            intent.intent = "document_query"
            if not intent.topic:
                intent.topic = safeguard_topic
            if not intent.question_type or intent.question_type == "social":
                intent.question_type = "topic_knowledge"
        else:
            print(f"[router] intent={intent.intent}", file=sys.stderr)
            try:
                answer = generate_general_chat_response(question, root_obs=root_obs)
                return {"answer": answer, "sources": []}
            except Exception as exc:
                print(f"[rag] General chat generation error: {exc}", file=sys.stderr)
                return {
                    "answer": "⚠️ The AI service is currently unavailable. Please try again in a moment.",
                    "sources": [],
                }
    else:
        print(f"[router] intent=document_query", file=sys.stderr)

    collection = get_chroma_collection(chat_id)

    try:
        doc_count = collection.count()
    except Exception:
        doc_count = 0

    if doc_count == 0:
        if root_obs is not None:
            root_obs.update(
                output={"answer": _NO_DOCUMENT_MSG, "sources": [], "status": "success"},
                metadata={
                    "mode": "document_qa",
                    "selected_sources": [],
                    "target_source": None,
                    "is_document_specific": False,
                    "status": "success",
                    "retrieved_sources": [],
                },
            )
        return {"answer": _NO_DOCUMENT_MSG, "sources": []}

    # ── Determine available sources and categorize images vs documents ──
    all_data = collection.get(include=["documents", "metadatas"])
    metadatas = all_data.get("metadatas", []) or []
    current_docs = list(all_data.get("documents") or [])
    current_metas = list(metadatas)

    image_sources: list[str] = []
    document_sources: list[str] = []
    seen_sources: set[str] = set()

    for m in metadatas:
        if not isinstance(m, dict):
            continue
        src = str(m.get("source", "")).strip()
        if not src or src in seen_sources:
            continue
        seen_sources.add(src)
        if m.get("type") == "image" or is_image_file(src):
            image_sources.append(src)
        else:
            document_sources.append(src)

    sources_found = image_sources + document_sources
    primary_source = sources_found[0] if sources_found else ""

    # Detect if query targets a specific document or image, or if selected_source was passed
    source_question = previous_user_question or working_question
    target_source = _detect_target_source(
        source_question,
        sources_found,
        explicit_source=selected_source,
        image_sources=image_sources,
        document_sources=document_sources,
    )
    is_doc_specific = bool(target_source or selected_source)
    if target_source:
        print(f"[rag] TARGET SOURCE SELECTED: '{target_source}' (explicit={selected_source})", file=sys.stderr)
        active_source = target_source
        file_label = f"uploaded file ({target_source})"
    else:
        active_source = primary_source
        if len(sources_found) > 1:
            file_label = f"uploaded files ({len(sources_found)} files)"
        elif primary_source:
            file_label = f"uploaded file ({primary_source})"
        else:
            file_label = "uploaded file"

    # ── Classify query mode ──
    mode = classify_query_intent(
        working_question,
        image_sources=image_sources,
        document_sources=document_sources,
        target_source=target_source,
    )

    # Deterministic visual-reference guard: figure/diagram/chart questions need
    # page-image inspection even if the probabilistic intent router underweights
    # the visual requirement. This is generic and does not enumerate questions.
    visual_term = re.search(r"\b(?:figure|fig\.?|diagram|chart|graph|illustration|flowchart|plot)\b", working_question, re.I)
    visual_action = re.search(
        r"\b(?:explain|describe|show|what(?:'s| is)?|tell me about|interpret|analy[sz]e|look at)\b",
        working_question,
        re.I,
    )
    has_structured_visual_ref = bool(re.search(r"\b(?:figure|fig\.?|diagram|chart|graph|illustration|flowchart|plot)\s*(?:no\.?|number)?\s*\d+\b", working_question, re.I))
    if visual_term and (visual_action or has_structured_visual_ref):
        try:
            intent.evidence_type = "VISUAL_EVIDENCE"
        except Exception:
            pass
        print("[router] deterministic visual-reference override", file=sys.stderr)

    # Named section routing is document-dependent. A generic request like
    # "Give me Lukasz Kaiser" may syntactically resemble a section request, but
    # it must only enter section mode when the target actually exists as a
    # heading in the uploaded document. Otherwise keep it in normal document QA.
    parsed_section_candidate = _parse_section_target(working_question)
    if parsed_section_candidate and parsed_section_candidate.get("title"):
        try:
            current_data = collection.get(include=["documents", "metadatas"])
            current_docs = list(current_data.get("documents") or [])
            current_metas = list(current_data.get("metadatas") or [])
            known_target = _generic_section_target_is_known(
                parsed_section_candidate, current_docs, current_metas, target_source=target_source
            )
            if known_target:
                mode = "section_extraction"
                print(
                    f"[router] document-derived section match: target={parsed_section_candidate.get('title')!r}",
                    file=sys.stderr,
                )
            elif mode == "section_extraction":
                mode = "document_qa"
                print(
                    f"[router] rejected generic section target {parsed_section_candidate.get('title')!r}; using document_qa",
                    file=sys.stderr,
                )
        except Exception as exc:
            if mode == "section_extraction" and not re.search(r"\b(?:section|chapter|part|step|item|point|heading)\b", working_question, re.I):
                mode = "document_qa"
            print(f"[router] section-target validation failed: {exc}; mode={mode}", file=sys.stderr)
    # A short follow-up inherits a previously identified section target. This is
    # deterministic so an intent/heading classifier cannot reinterpret "remaining"
    # as a new heading-list request.
    if previous_user_question and _parse_section_target(previous_user_question):
        prev_parse = _parse_section_target(previous_user_question)
        prev_explicit = bool(re.search(r"\b(?:section|chapter|part|step|item|point|heading)\b", previous_user_question, re.I))
        prev_known = _generic_section_target_is_known(
            prev_parse, current_docs, current_metas, target_source=target_source
        ) if prev_parse else False
        if prev_explicit or prev_known:
            mode = "section_extraction"
            print("[conversation] inherited previous section target for follow-up", file=sys.stderr)

    if root_obs is not None:
        root_obs.update(
            metadata={
                "mode": mode,
                "selected_sources": sources_found,
                "target_source": target_source,
                "is_document_specific": is_doc_specific,
                "image_sources": image_sources,
                "document_sources": document_sources,
            }
        )

    # ── GENERIC MULTIMODAL VISUAL QA ──────────────────────────────
    # If question depends on visual appearance, layout, spatial position, handwriting,
    # circled/marked elements, diagrams, or charts, inspect the original page image(s).
    if getattr(intent, "evidence_type", None) in ("VISUAL_EVIDENCE", "BOTH"):
        figure_ref_present = bool(re.search(
            r"\b(?:figure|fig\.?|diagram|chart|graph|illustration|flowchart|plot)\s*(?:no\.?|number)?\s*\d+\b",
            question,
            re.IGNORECASE,
        ))
        candidate_pages = _locate_candidate_pages(
            collection,
            question,
            chat_id=chat_id,
            target_source=target_source,
            max_pages=2 if figure_ref_present else 5,
        )
        if candidate_pages:
            img_paths = [cp["image_path"] for cp in candidate_pages]
            src_label = candidate_pages[0]["source"]
            ans = generate_visual_qa_response(
                question=question,
                image_paths=img_paths,
                source_label=src_label,
                root_obs=root_obs,
            )
            sources_out = [
                {
                    "label": f"{cp['source']} (Page {cp['page']})",
                    "snippet": f"Visual evidence inspected from page {cp['page']}",
                }
                for cp in candidate_pages
            ]
            if root_obs is not None:
                try:
                    root_obs.update(
                        output={"answer": ans, "sources": sources_out, "status": "success"},
                        metadata={
                            "mode": "visual_qa",
                            "evidence_type": intent.evidence_type,
                            "candidate_pages": [(cp["source"], cp["page"]) for cp in candidate_pages],
                        },
                    )
                except Exception:
                    pass
            return {"answer": ans, "sources": sources_out}

    # ── IMAGE QA MODE ───────────────────────────────────────────
    if mode == "image_qa":
        def _run_image_retrieval():
            img_docs, img_dists, img_metas = _retrieve_image_chunks(
                collection,
                target_source=target_source,
                image_sources=image_sources,
                question=question,
            )
            stats = {
                "candidate_count": len(img_docs),
                "filtered_count": len(img_docs),
                "dedup_count": len(img_docs),
                "reranked_count": len(img_docs),
                "final_chunk_count": len(img_docs),
                "distances": img_dists,
                "sources": _safe_source_list(img_metas),
            }
            return (img_docs, img_dists, img_metas), stats

        (img_docs, img_dists, img_metas), ret_lat_ms = _langfuse_track_retrieval(
            question, sources_found, target_source, doc_count,
            _run_image_retrieval,
            root_obs=root_obs,
            source_func=_retrieve_image_chunks,
        )

        if not img_docs:
            not_found_ans = "I could not find any readable content from the uploaded image(s)."
            if root_obs is not None:
                root_obs.update(
                    metadata={
                        "mode": "image_qa",
                        "retrieved_sources": [],
                        "retrieval_latency_ms": ret_lat_ms,
                        "status": "success",
                    },
                    output={"answer": not_found_ans, "sources": [], "mode": "image_qa", "status": "success"},
                )
            return {"answer": not_found_ans, "sources": []}

        clean_context, sources_out = _build_image_context(
            img_docs,
            img_metas,
            target_source=target_source,
            image_sources=image_sources,
            max_chars=MAX_CONTEXT_CHARS,
        )

        if not clean_context.strip():
            not_found_ans = "I could not extract any readable content from the uploaded image(s)."
            return {"answer": not_found_ans, "sources": sources_out}

        if _groq_client is None:
            return {
                "answer": (
                    "⚠️ The AI backend is not configured. "
                    "Please set GROQ_API_KEY in backend/.env and restart the server."
                ),
                "sources": [],
            }

        compiled_prompt, prompt_client = _get_managed_prompt(
            "ocr/querymind-image-qa",
            context=clean_context,
            question=question,
        )

        if "IMAGE CONTENT:" in compiled_prompt:
            parts = compiled_prompt.split("IMAGE CONTENT:", 1)
            system_prompt = parts[0].strip()
            user_message = "IMAGE CONTENT:" + parts[1]
        else:
            system_prompt = compiled_prompt
            user_message = f"IMAGE CONTENT:\n{clean_context}\n\nUSER QUESTION:\n{question}"

        print(
            f"[rag] IMAGE QA MODE → Groq: context_chars={len(clean_context)}, "
            f"chunks={len(img_docs)}, prompt=ocr/querymind-image-qa",
            file=sys.stderr,
        )

        try:
            response = _langfuse_track_generation(
                "image_qa",
                system_prompt,
                user_message,
                question=question,
                sources=_safe_source_list(img_metas),
                context_size=len(clean_context),
                context_chunks=len(sources_out),
                prompt_client=prompt_client,
                root_obs=root_obs,
            )
            answer = response.choices[0].message.content
            if answer and answer.strip():
                print(f"[rag] IMAGE QA ANSWER: {answer.strip()[:200]}", file=sys.stderr)
                return {"answer": answer.strip(), "sources": sources_out}
            return {"answer": "I could not extract that information from the image.", "sources": sources_out}
        except Exception as exc:
            print(f"[rag] Groq error in image_qa mode: {exc}", file=sys.stderr)
            return {
                "answer": (
                    "⚠️ The AI service is currently unavailable. "
                    "Please try again in a moment or rephrase your question."
                ),
                "sources": [],
            }

    # ── DOCUMENT-WIDE REFERENCE/BIBLIOGRAPHY MODES ───────────────────────
    # These operations must inspect the complete source document instead of
    # relying on top-k retrieval, because a bibliography can span many pages.
    if mode in {"reference_count", "reference_list"}:
        print(f"[router] mode={mode}", file=sys.stderr)

        source_names: list[str] = []
        seen_sources: set[str] = set()
        for meta in current_metas:
            if not isinstance(meta, dict):
                continue
            src = str(meta.get("source", "")).strip()
            if src and src not in seen_sources:
                seen_sources.add(src)
                source_names.append(src)
        if target_source:
            source_names = [
                s for s in source_names
                if os.path.basename(s).casefold() == os.path.basename(target_source).casefold()
            ] or [target_source]

        results: list[tuple[str, list[str]]] = []
        for source_name in source_names:
            source_text, _source_path = _load_original_source_text(chat_id, source_name, user_email=user_email)
            if not source_text:
                continue
            try:
                refs = extract_reference_entries(source_text)
            except Exception as exc:
                print(f"[structure] reference extraction failed for '{source_name}': {exc}", file=sys.stderr)
                refs = []
            results.append((source_name, refs))

        if not results:
            return {"answer": _not_covered_msg(file_label), "sources": []}

        if mode == "reference_count":
            nonempty = [(src, refs) for src, refs in results if refs]
            if not nonempty:
                return {
                    "answer": f"I could not identify a numbered reference/bibliography list in the {file_label}.",
                    "sources": [],
                }
            total = sum(len(refs) for _, refs in nonempty)
            if len(nonempty) == 1:
                src, refs = nonempty[0]
                answer = f"There are {len(refs)} numbered reference entries in the document."
            else:
                lines = [f"There are {total} numbered reference entries across the uploaded documents."]
                lines.extend(f"- {src}: {len(refs)} references" for src, refs in nonempty)
                answer = "\n".join(lines)
            print(
                f"[references] deterministic_count={total} sources={len(nonempty)}",
                file=sys.stderr,
            )
            return {"answer": answer, "sources": [{"label": src} for src, _ in nonempty]}

        # reference_list
        nonempty = [(src, refs) for src, refs in results if refs]
        if not nonempty:
            return {
                "answer": f"I could not identify a numbered reference/bibliography list in the {file_label}.",
                "sources": [],
            }
        parts = ["### References"]
        for src, refs in nonempty:
            parts.append(f"\n**{src}**")
            parts.extend(f"- {entry}" for entry in refs)
        answer = "\n".join(parts)
        return {"answer": answer, "sources": [{"label": src} for src, _ in nonempty]}

    # ── FULL-DOCUMENT BATCHED SUMMARY MODE (Part 1 / Section 1) ──
    if mode == "summary":
        print(f"[router] mode=summary", file=sys.stderr)
        try:
            raw_data = collection.get(include=["documents", "metadatas"])
            all_docs = list(raw_data.get("documents") or [])
            all_metas = list(raw_data.get("metadatas") or [])
            all_ids = list(raw_data.get("ids") or [])
        except Exception as exc:
            print(f"[summary] Collection get failed: {exc}", file=sys.stderr)
            all_docs, all_metas, all_ids = [], [], []

        scoped_rows = []
        norm_target = os.path.basename(target_source.strip()).lower() if target_source else None
        for idx, (doc, meta) in enumerate(zip(all_docs, all_metas or [{}] * len(all_docs))):
            src = str((meta or {}).get("source", "")).strip()
            norm_src = os.path.basename(src).strip().lower()
            if norm_target and norm_src != norm_target:
                continue
            chunk_id = str(all_ids[idx]) if idx < len(all_ids) else str(idx)
            scoped_rows.append((idx, chunk_id, doc or "", meta or {}))

        def _chunk_sort_key(row):
            orig_idx, cid, _, _ = row
            m = re.search(r"_(\d+)$", cid)
            if m:
                return (0, int(m.group(1)))
            return (1, orig_idx)

        scoped_rows.sort(key=_chunk_sort_key)
        scoped_docs = [r[2] for r in scoped_rows]
        scoped_metas = [r[3] for r in scoped_rows]
        sources_out = [{"label": s} for s in _safe_source_list(scoped_metas)]

        total_chunks = len(scoped_docs)
        total_chars = sum(len(d) for d in scoped_docs)
        print(f"[summary] full_document_scope=true", file=sys.stderr)
        print(f"[summary] total_chunks={total_chunks}", file=sys.stderr)

        if not scoped_docs:
            return {"answer": _not_covered_msg(file_label), "sources": []}

        system_prompt, prompt_client = _get_managed_prompt(
            "rag/querymind-summary",
            file_label=file_label,
        )

        # Single-pass for small/medium documents
        if total_chars <= 12000:
            print(f"[summary] batch_count=1", file=sys.stderr)
            print(f"[summary] batch=1/1", file=sys.stderr)
            full_context = "\n\n".join(scoped_docs)
            user_message = f"Document content ({file_label}):\n\n{full_context}\n\nUser request: {question}"

            try:
                response = _langfuse_track_generation(
                    "summary",
                    system_prompt,
                    user_message,
                    question=question,
                    sources=sources_out,
                    context_size=len(full_context),
                    context_chunks=total_chunks,
                    prompt_client=prompt_client,
                    root_obs=root_obs,
                )
                answer = (response.choices[0].message.content or "").strip()
                print(f"[summary] final_summary_complete=true", file=sys.stderr)
                return {"answer": answer or _not_covered_msg(file_label), "sources": sources_out}
            except Exception as exc:
                print(f"[summary] Single-pass error: {exc}", file=sys.stderr)
                return {
                    "answer": "⚠️ The AI service is currently unavailable. Please try again in a moment.",
                    "sources": sources_out,
                }

        # Multi-pass sequential batching + synthesis for large documents
        batch_limit_chars = 9000
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_batch_len = 0

        for ch in scoped_docs:
            if current_batch_len + len(ch) > batch_limit_chars and current_batch:
                batches.append(current_batch)
                current_batch = [ch]
                current_batch_len = len(ch)
            else:
                current_batch.append(ch)
                current_batch_len += len(ch)
        if current_batch:
            batches.append(current_batch)

        batch_count = len(batches)
        print(f"[summary] batch_count={batch_count}", file=sys.stderr)

        batch_summaries: list[str] = []
        for b_idx, batch_chunks in enumerate(batches, 1):
            print(f"[summary] batch={b_idx}/{batch_count}", file=sys.stderr)
            b_text = "\n\n".join(batch_chunks)
            b_user_msg = (
                f"Document part {b_idx} of {batch_count} ({file_label}):\n\n"
                f"{b_text}\n\n"
                f"Summarize the key information, findings, metrics, and concepts from this section."
            )
            try:
                res = _langfuse_track_generation(
                    "summary",
                    system_prompt,
                    b_user_msg,
                    question=question,
                    sources=sources_out,
                    context_size=len(b_text),
                    context_chunks=len(batch_chunks),
                    prompt_client=prompt_client,
                    root_obs=root_obs,
                )
                b_ans = (res.choices[0].message.content or "").strip()
                if b_ans:
                    batch_summaries.append(f"Section Summary (Part {b_idx}):\n{b_ans}")
            except Exception as exc:
                print(f"[summary] Batch {b_idx} failed: {exc}", file=sys.stderr)

        if not batch_summaries:
            return {"answer": _not_covered_msg(file_label), "sources": sources_out}

        synthesis_input = "\n\n".join(batch_summaries)
        synthesis_msg = (
            f"Below are summaries covering the COMPLETE document {file_label} across all {batch_count} sequential parts:\n\n"
            f"{synthesis_input}\n\n"
            f"User request: {question}\n\n"
            f"Synthesize a coherent, unified, and comprehensive final summary representing the ENTIRE document."
        )

        try:
            final_res = _langfuse_track_generation(
                "summary",
                system_prompt,
                synthesis_msg,
                question=question,
                sources=sources_out,
                context_size=len(synthesis_input),
                context_chunks=batch_count,
                prompt_client=prompt_client,
                root_obs=root_obs,
            )
            final_answer = (final_res.choices[0].message.content or "").strip()
            print(f"[summary] final_summary_complete=true", file=sys.stderr)
            return {"answer": final_answer or _not_covered_msg(file_label), "sources": sources_out}
        except Exception as exc:
            print(f"[summary] Final synthesis failed: {exc}", file=sys.stderr)
            return {"answer": "\n\n".join(batch_summaries), "sources": sources_out}

    def _find_exact_numbered_heading(source_text: str, requested_num: str) -> Optional[str]:
        """Find one exact numbered heading, including headings split across PDF lines."""
        wanted = str(requested_num or "").strip()
        if not re.fullmatch(r"\d+(?:\.\d+)*", wanted):
            return None

        for _line_idx, number, title in _iter_numbered_heading_candidates(source_text or ""):
            if number == wanted:
                cleaned = _clean_heading_candidate(title)
                if cleaned:
                    return f"{wanted}. {cleaned}"
        return None


    def _clean_heading_candidate(value: str) -> Optional[str]:
        title = re.sub(r"\s+", " ", (value or "").strip(" \t:;–—-"))
        if not title or len(title) > 180 or title[0].isdigit():
            return None
        if title.lower().startswith("reference table"):
            return None
        first_alpha = next((ch for ch in title if ch.isalpha()), "")
        if first_alpha and first_alpha.islower():
            return None
        # Reject obvious prose fragments.
        if len(title.split()) > 18:
            return None
        if re.search(r"[.!?]\s+[A-Z]", title):
            return None
        return title

    def _heading_from_collection(requested_num: str, docs: list[str], metas: list[dict]) -> Optional[str]:
        """Find a specific numbered heading from current index metadata/text."""
        wanted = str(requested_num).strip()
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            if str(meta.get("section_number", "") or "").strip() == wanted:
                title = str(meta.get("section_title", "") or "").strip()
                if title:
                    return f"{wanted}. {title}"
        for doc in docs:
            found = _find_exact_numbered_heading(doc or "", wanted)
            if found:
                return found
        return None

    # ── SPECIFIC NUMBERED HEADING LOOKUP MODE ─────────────────
    # A singular request such as "Tell me heading 13" must never return the
    # entire heading list. Read the original source structure and return only
    # the requested heading title.
    if mode == "heading_lookup":
        print("[router] mode=heading_lookup", file=sys.stderr)
        parsed = _parse_section_target(working_question)
        requested_num = parsed.get("number") if parsed else None
        if not requested_num:
            return {"answer": "I could not determine which heading number you requested.", "sources": []}

        try:
            raw_data = collection.get(include=["documents", "metadatas"])
            all_docs = list(raw_data.get("documents") or [])
            all_metas = list(raw_data.get("metadatas") or [])
        except Exception as exc:
            print(f"[structure] Collection get failed: {exc}", file=sys.stderr)
            all_docs, all_metas = [], []

        source_names: list[str] = []
        seen_source_names: set[str] = set()
        for meta in all_metas:
            if not isinstance(meta, dict):
                continue
            src = str(meta.get("source", "")).strip()
            if src and src not in seen_source_names:
                seen_source_names.add(src)
                source_names.append(src)
        if target_source:
            source_names = [s for s in source_names if os.path.basename(s).casefold() == os.path.basename(target_source).casefold()] or [target_source]

        matches: list[tuple[str, str]] = []
        for source_name in source_names:
            headings: list[str] = []

            # First use the exact section metadata already stored in the current
            # chat index. This works even when local source files were removed
            # after upload/restart.
            scoped_docs = []
            scoped_metas = []
            for doc, meta in zip(all_docs, all_metas or [{}] * len(all_docs)):
                meta = meta or {}
                if os.path.basename(str(meta.get("source", ""))).casefold() != os.path.basename(source_name).casefold():
                    continue
                scoped_docs.append(doc or "")
                scoped_metas.append(meta)
            exact_collection = _heading_from_collection(str(requested_num), scoped_docs, scoped_metas)
            if exact_collection:
                headings = [exact_collection]

            # Fall back to the original file from local storage, then Supabase.
            if not headings:
                source_text, source_path = _load_original_source_text(chat_id, source_name, user_email=user_email)
                if source_text:
                    try:
                        exact = _find_exact_numbered_heading(source_text, str(requested_num))
                        if exact:
                            headings = [exact]
                        else:
                            pdf_records = extract_numbered_headings_from_pdf_file(source_path) if source_path and source_name.lower().endswith(".pdf") else []
                            exact_pdf = next((r.full_text for r in pdf_records if r.number == str(requested_num)), None)
                            headings = [exact_pdf] if exact_pdf else extract_headings_from_source_text(source_text)
                    except Exception as exc:
                        print(f"[structure] heading lookup source extraction failed for '{source_name}': {exc}", file=sys.stderr)

            if not headings:
                headings = _extract_document_headings(scoped_docs)

            for heading in headings:
                m = re.match(r"^\s*(\d+(?:\.\d+)*)\.\s+", heading)
                if m and m.group(1) == str(requested_num):
                    matches.append((source_name, heading))
                    break

        if not matches:
            return {
                "answer": f"Heading {requested_num} does not appear in the {file_label}.",
                "sources": [],
            }

        if len(matches) == 1:
            answer = matches[0][1]
        else:
            answer = "### Matching Heading\n\n" + "\n".join(f"- **{src}**: {heading}" for src, heading in matches)
        return {"answer": answer, "sources": [{"label": src} for src, _ in matches]}

    # ── GLOBAL STRUCTURAL HEADING EXTRACTION MODE ────────────────
    if mode == "heading_extraction":
        print("[router] mode=heading_extraction", file=sys.stderr)

        # Heading extraction is structural, so read the original uploaded file
        # whenever it is available. This prevents stale/legacy vector chunks from
        # inventing false headings or losing numbering.
        try:
            raw_data = collection.get(include=["documents", "metadatas"])
            all_docs = list(raw_data.get("documents") or [])
            all_metas = list(raw_data.get("metadatas") or [])
        except Exception as exc:
            print(f"[structure] Collection get failed: {exc}", file=sys.stderr)
            all_docs, all_metas = [], []

        source_names: list[str] = []
        seen_source_names: set[str] = set()
        for meta in all_metas:
            if not isinstance(meta, dict):
                continue
            src = str(meta.get("source", "")).strip()
            if src and src not in seen_source_names:
                seen_source_names.add(src)
                source_names.append(src)

        if target_source:
            source_names = [s for s in source_names if os.path.basename(s).casefold() == os.path.basename(target_source).casefold()] or [target_source]

        headings_by_source: list[tuple[str, list[str]]] = []
        for source_name in source_names:
            source_headings: list[str] = []
            source_text, source_path = _load_original_source_text(chat_id, source_name, user_email=user_email)
            if source_text:
                try:
                    source_headings = extract_headings_from_source_text(source_text)
                    print(
                        f"[structure] source-first headings: source='{source_name}' count={len(source_headings)}",
                        file=sys.stderr,
                    )
                except Exception as exc:
                    print(f"[structure] source heading extraction failed for '{source_name}': {exc}", file=sys.stderr)

            if not source_headings:
                fallback_docs = [
                    doc or ""
                    for doc, meta in zip(all_docs, all_metas or [{}] * len(all_docs))
                    if not target_source or os.path.basename(str((meta or {}).get("source", ""))).casefold() == os.path.basename(source_name).casefold()
                ]
                source_headings = _extract_document_headings(fallback_docs)
            if source_headings:
                headings_by_source.append((source_name, source_headings))

        # If source lookup is unavailable, retain the previous metadata/chunk fallback.
        if not headings_by_source and all_docs:
            fallback_docs = []
            fallback_metas = []
            norm_target = os.path.basename(target_source.strip()).casefold() if target_source else None
            for doc, meta in zip(all_docs, all_metas or [{}] * len(all_docs)):
                meta = meta or {}
                src = str(meta.get("source", "")).strip()
                if norm_target and os.path.basename(src).casefold() != norm_target:
                    continue
                fallback_docs.append(doc or "")
                fallback_metas.append(meta)
            headings = _extract_document_headings(fallback_docs)
            if headings:
                headings_by_source.append((target_source or "uploaded document", headings))

        if not headings_by_source:
            return {"answer": _not_covered_msg(file_label), "sources": []}

        # One document keeps the clean historical output. Multiple documents are
        # grouped by source so headings never become ambiguous across files.
        if len(headings_by_source) == 1:
            headings = headings_by_source[0][1]
            answer = "### Document Headings\n\n" + "\n".join(f"- {h}" for h in headings)
        else:
            parts = ["### Document Headings"]
            for source_name, headings in headings_by_source:
                parts.append(f"\n**{source_name}**\n" + "\n".join(f"- {h}" for h in headings))
            answer = "\n".join(parts)

        sources_out = [{"label": s} for s, _ in headings_by_source]
        print(
            f"[structure] deterministic_headings={sum(len(h) for _, h in headings_by_source)} sources={len(headings_by_source)}",
            file=sys.stderr,
        )
        return {"answer": answer, "sources": sources_out}

    def _extract_requested_section_from_source(source_text: str, requested_number: Optional[str], requested_title: Optional[str]) -> tuple[str, str] | tuple[None, None]:
        """Compatibility wrapper around the shared structural section extractor."""
        content, heading = extract_numbered_section(
            source_text,
            requested_number=requested_number,
            requested_title=requested_title,
        )
        return content, heading


    # ── NAMED SECTION EXTRACTION MODE ───────────────────────────
    if mode == "section_extraction":
        print("[router] mode=section_extraction", file=sys.stderr)
        section_query = previous_user_question if (previous_user_question and _parse_section_target(previous_user_question)) else working_question
        parsed_target = _parse_section_target(section_query)
        req_num = parsed_target.get("number") if parsed_target else None
        req_title = parsed_target.get("title") if parsed_target else None
        if not req_num and not req_title:
            req_title = _extract_target_section(section_query) or section_query

        docs, distances, metas = _retrieve_named_section(
            collection,
            target_source=target_source,
            section_number=req_num,
            section_title=req_title,
        )
        if not docs:
            # Chroma may contain an older index or no section metadata at all.
            # Fall back to the original document itself (local cache -> Supabase).
            candidate_sources = [target_source] if target_source else []
            if not candidate_sources:
                try:
                    data = collection.get(include=["metadatas"])
                    for meta in data.get("metadatas") or []:
                        src = str((meta or {}).get("source", "")).strip()
                        if src and os.path.basename(src).casefold() not in {os.path.basename(x).casefold() for x in candidate_sources}:
                            candidate_sources.append(src)
                except Exception:
                    candidate_sources = []

            for source_name in candidate_sources:
                source_text, _source_path = _load_original_source_text(chat_id, source_name, user_email=user_email)
                if not source_text:
                    continue
                direct_content, direct_heading = _extract_requested_section_from_source(
                    source_text, req_num, req_title
                )
                if direct_content:
                    sources_out = [{"label": source_name}]
                    print(
                        f"[section] source-direct fallback: heading={direct_heading!r} chars={len(direct_content)}",
                        file=sys.stderr,
                    )
                    if "explain" in question.lower() and not _is_full_section_request(question):
                        try:
                            system_prompt, prompt_client = _get_managed_prompt(
                                "rag/querymind-section-extraction",
                                file_label=file_label,
                                section_name=direct_heading or req_title or req_num or "requested section",
                            )
                            response = _langfuse_track_generation(
                                "section_extraction",
                                system_prompt,
                                f"Complete source section from {file_label}:\n\n{direct_content}\n\nUser request: {question}",
                                question=question,
                                sources=[source_name],
                                context_size=len(direct_content),
                                context_chunks=1,
                                prompt_client=prompt_client,
                                root_obs=root_obs,
                            )
                            answer = (response.choices[0].message.content or "").strip()
                            return {"answer": answer or direct_content, "sources": sources_out}
                        except Exception as exc:
                            print(f"[section] source-direct generation failed: {exc}; returning source content", file=sys.stderr)
                    return {"answer": direct_content, "sources": sources_out}

            label = req_title or req_num or "requested section"
            return {
                "answer": f"Section '{label}' does not appear in the {file_label}.",
                "sources": [],
            }

        # Each section chunk carries its exact heading as the first line. Remove
        # repeated internal headers while preserving every body chunk in order.
        section_lines: list[str] = []
        full_heading = None
        if metas:
            first_meta = metas[0] or {}
            if first_meta.get("section_number") and first_meta.get("section_title"):
                full_heading = f"{first_meta['section_number']}. {first_meta['section_title']}"
        for idx, doc in enumerate(docs):
            lines = (doc or "").splitlines()
            heading_title = None
            if lines and re.match(r"^\d+(?:\.\d+)*\.\s+.+$", lines[0].strip()):
                if full_heading is None:
                    full_heading = lines[0].strip()
                heading_title = re.sub(r"^\d+(?:\.\d+)*\.\s+", "", lines[0].strip())
                lines = lines[1:]
            # Structure-aware chunks already carry the section title as their first
            # body line. Remove that duplicate so the user sees the title once.
            if heading_title and lines and re.sub(r"\s+", " ", lines[0].strip()).casefold() == re.sub(r"\s+", " ", heading_title).casefold():
                lines = lines[1:]
            text_part = "\n".join(lines).strip()
            if text_part:
                section_lines.append(text_part)

        body = "\n\n".join(section_lines).strip()
        display = full_heading or req_title or req_num or "Requested Section"
        extracted_content = f"{display}\n\n{body}".strip()
        sources_out = [{"label": s} for s in _safe_source_list(metas)]
        print(f"[section] complete retrieval: heading='{display}' chunks={len(docs)} chars={len(extracted_content)}", file=sys.stderr)

        # A named-section request is inherently a completeness request. Returning
        # the complete extracted section is safer than asking an LLM to summarize
        # it and accidentally omit paragraphs/tables.
        if _is_full_section_request(question) or "explain" not in question.lower():
            return {"answer": extracted_content, "sources": sources_out}

        system_prompt, prompt_client = _get_managed_prompt(
            "rag/querymind-section-extraction",
            file_label=file_label,
            section_name=display,
        )
        user_message = (
            f"Complete source section from {file_label}:\n\n{extracted_content}\n\n"
            f"User request: {question}\n\n"
            "Explain this section accurately using all supplied content. Do not omit important details, "
            "do not invent information, and do not rewrite section numbers."
        )
        try:
            response = _langfuse_track_generation(
                "section_extraction", system_prompt, user_message,
                question=question, sources=_safe_source_list(metas),
                context_size=len(extracted_content), context_chunks=len(docs),
                prompt_client=prompt_client, root_obs=root_obs,
            )
            answer = (response.choices[0].message.content or "").strip()
            return {"answer": answer or extracted_content, "sources": sources_out}
        except Exception as exc:
            print(f"[section] Generation error: {exc}; returning complete source section", file=sys.stderr)
            return {"answer": extracted_content, "sources": sources_out}

    # ── SPECIAL DEFINITION MODE ──────────────────────────────────
    definition_term = _extract_definition_term(question) if mode == "definition" else None
    definition_mode = definition_term is not None
    topic_term = _extract_topic_term(question) if mode == "topic" else None

    if definition_mode:
        def _run_def_retrieval():
            term_found, term_docs, term_metas, verified_candidate = _find_term_in_entire_document(
                chat_id, collection, definition_term, target_source=target_source
            )
            docs = term_docs[:RERANK_TOP_N]
            metas = term_metas[:RERANK_TOP_N]
            distances = [0.0] * len(docs)
            stats = {
                "candidate_count": doc_count,
                "filtered_count": len(term_docs),
                "dedup_count": len(term_docs),
                "reranked_count": len(docs),
                "final_chunk_count": len(docs),
                "distances": distances,
                "sources": _safe_source_list(metas),
                "term_found": bool(term_found),
                "term": definition_term,
            }
            return (docs, distances, metas, term_found, verified_candidate), stats

        (docs, distances, metas, term_found, verified_candidate), ret_lat_ms = _langfuse_track_retrieval(
            question, sources_found, target_source, doc_count,
            _run_def_retrieval,
            root_obs=root_obs,
        )

        if not term_found:
            if root_obs is not None:
                root_obs.update(
                    metadata={
                        "mode": "definition",
                        "term": definition_term,
                        "term_found": False,
                        "retrieved_sources": [],
                        "retrieval_latency_ms": ret_lat_ms,
                        "status": "success",
                    },
                    output={
                        "answer": _not_covered_msg(file_label),
                        "sources": [],
                        "mode": "definition",
                        "status": "success",
                    },
                )
            return {"answer": _not_covered_msg(file_label), "sources": []}

        term_to_define = verified_candidate or definition_term
        context, sources_out = _build_bounded_context(
            docs, metas, active_source, max_chars=MAX_CONTEXT_CHARS
        )

        if _groq_client is None:
            return {
                "answer": (
                    "⚠️ The AI backend is not configured. "
                    "Please set GROQ_API_KEY in backend/.env and restart the server."
                ),
                "sources": [],
            }

        system_prompt, prompt_client = _get_managed_prompt(
            "rag/querymind-definition",
            file_label=file_label,
            term=term_to_define,
        )
        user_message = (
            f"Document evidence proving the term is present:\n{context}\n\n"
            f"Verified term to define: {term_to_define}\n"
            f"Question: {question}"
        )

        print(
            f"[rag] DEFINITION MODE → Groq: term='{term_to_define}' (from query '{definition_term}'), "
            f"evidence_chunks={len(sources_out)}, context_len={len(context)}, temperature=0",
            file=sys.stderr,
        )

        try:
            response = _langfuse_track_generation(
                "definition", system_prompt, user_message,
                question=question,
                sources=_safe_source_list(metas),
                context_size=len(context),
                context_chunks=len(sources_out),
                prompt_client=prompt_client,
                root_obs=root_obs,
            )
            answer = response.choices[0].message.content
            if answer and answer.strip():
                print(f"[rag] DEFINITION ANSWER: {answer.strip()[:200]}", file=sys.stderr)
                return {"answer": answer.strip(), "sources": sources_out}
            return {"answer": _not_covered_msg(file_label), "sources": sources_out}
        except Exception as exc:
            print(f"[rag] Groq error in definition mode: {exc}", file=sys.stderr)
            return {
                "answer": (
                    "⚠️ The AI service is currently unavailable. "
                    "Please try again in a moment or rephrase your question."
                ),
                "sources": [],
            }

    # ── CONTROLLED TOPIC-EXPLANATION MODE ───────────────────────
    if topic_term:
        def _run_topic_retrieval():
            topic_found, topic_docs, topic_metas, matched_by = _find_topic_in_entire_document(
                collection, topic_term, target_source=target_source
            )
            docs = topic_docs[:RERANK_TOP_N]
            metas = topic_metas[:RERANK_TOP_N]
            distances = [0.0] * len(docs)
            stats = {
                "candidate_count": doc_count,
                "filtered_count": len(topic_docs),
                "dedup_count": len(topic_docs),
                "reranked_count": len(docs),
                "final_chunk_count": len(docs),
                "distances": distances,
                "sources": _safe_source_list(metas),
                "topic_found": bool(topic_found),
                "topic": topic_term,
            }
            return (docs, distances, metas, topic_found), stats

        (docs, distances, metas, topic_found), ret_lat_ms = _langfuse_track_retrieval(
            question, sources_found, target_source, doc_count,
            _run_topic_retrieval,
            root_obs=root_obs,
            source_func=_find_topic_in_entire_document,
        )

        if not topic_found:
            if root_obs is not None:
                root_obs.update(
                    metadata={
                        "mode": "topic",
                        "topic": topic_term,
                        "topic_found": False,
                        "retrieved_sources": [],
                        "retrieval_latency_ms": ret_lat_ms,
                        "status": "success",
                    },
                    output={
                        "answer": _not_covered_msg(file_label),
                        "sources": [],
                        "mode": "topic",
                        "status": "success",
                    },
                )
            return {"answer": _not_covered_msg(file_label), "sources": []}

        context, sources_out = _build_bounded_context(
            docs, metas, active_source, max_chars=MAX_CONTEXT_CHARS
        )

        if _groq_client is None:
            return {
                "answer": (
                    "⚠️ The AI backend is not configured. "
                    "Please set GROQ_API_KEY in backend/.env and restart the server."
                ),
                "sources": [],
            }

        system_prompt, prompt_client = _get_managed_prompt(
            "rag/querymind-topic",
            file_label=file_label,
            topic=topic_term,
        )
        user_message = (
            f"Document evidence showing that the topic is present:\n{context}\n\n"
            f"Verified topic: {topic_term}\nQuestion: {question}"
        )
        print(
            f"[rag] TOPIC MODE → Groq: topic='{topic_term}', context_len={len(context)}",
            file=sys.stderr,
        )
        try:
            response = _langfuse_track_generation(
                "topic", system_prompt, user_message,
                question=question,
                sources=_safe_source_list(metas),
                context_size=len(context),
                context_chunks=len(sources_out),
                prompt_client=prompt_client,
                root_obs=root_obs,
            )
            answer = response.choices[0].message.content
            if answer and answer.strip():
                return {"answer": answer.strip(), "sources": sources_out}
            return {"answer": _not_covered_msg(file_label), "sources": sources_out}
        except Exception as exc:
            print(f"[rag] Groq error in topic mode: {exc}", file=sys.stderr)
            return {
                "answer": (
                    "⚠️ The AI service is currently unavailable. "
                    "Please try again in a moment or rephrase your question."
                ),
                "sources": [],
            }


    # ── UNIVERSAL TOPIC GATE (document_qa mode) ──────────────────────────────
    # Two-stage grounding system:
    #
    #   Stage 1 — Topic extraction:
    #     Use the topic extracted by the LLM intent router (semantic, handles all
    #     patterns: "Tell me the name of X", "Who works as X?", "AI/ML intern",
    #     "color of biryani", etc.).  Fall back to the regex extractor when the
    #     router did not return a topic (e.g., old model output, parsing error).
    #
    #   Stage 2 — Topic verification + routing:
    #     • topic NOT in document  → not-covered response (no general knowledge)
    #     • topic IS in document + question_type=topic_knowledge
    #                              → topic-mode generation (general knowledge allowed
    #                                about the VERIFIED topic)
    #     • topic IS in document + question_type=document_fact (or None)
    #                              → fall through to RAG (answer from document)
    #     • no topic extracted     → fall through to RAG (safe default)
    #
    _llm_topic: Optional[str] = getattr(intent, "topic", None)
    _llm_question_type: Optional[str] = getattr(intent, "question_type", None)
    _info_topic: Optional[str] = _llm_topic or _extract_informational_topic(working_question)

    if _info_topic:
        print(f"[router] intent=document_query", file=sys.stderr)
        print(f"[topic] extracted='{_info_topic}' (source={'llm-router' if _llm_topic else 'regex-fallback'})", file=sys.stderr)

        _topic_found, _topic_docs_gate, _topic_metas_gate, _topic_matched_by = (
            _find_topic_in_entire_document(
                collection, _info_topic, target_source=target_source
            )
        )

        if _topic_found:
            _verified_src = (
                (_topic_metas_gate[0] or {}).get("source", "")
                if _topic_metas_gate else ""
            )
            _knowledge_allowed = _llm_question_type == "topic_knowledge"
            print(f"[topic] verified=True matched_by='{_topic_matched_by}' source='{_verified_src}'", file=sys.stderr)
            print(f"[topic] knowledge_allowed={_knowledge_allowed} question_type={_llm_question_type!r}", file=sys.stderr)

            # ── TOPIC KNOWLEDGE PATH ──────────────────────────────────────────
            # The topic exists in the document and the question asks about general
            # properties/attributes of that topic (color, ingredients, history, etc.).
            # Use topic-mode generation so the LLM can answer from general knowledge
            # about the VERIFIED topic, with document evidence provided as context.
            if _knowledge_allowed:
                _gen_docs = _topic_docs_gate[:RERANK_TOP_N]
                _gen_metas = _topic_metas_gate[:RERANK_TOP_N]
                context, sources_out = _build_bounded_context(
                    _gen_docs, _gen_metas, active_source, max_chars=MAX_CONTEXT_CHARS
                )
                if _groq_client is None:
                    return {
                        "answer": (
                            "⚠️ The AI backend is not configured. "
                            "Please set GROQ_API_KEY in backend/.env and restart the server."
                        ),
                        "sources": [],
                    }
                system_prompt, prompt_client = _get_managed_prompt(
                    "rag/querymind-topic",
                    file_label=file_label,
                    topic=_info_topic,
                )
                user_message = (
                    f"Document evidence showing that the topic is present:\n{context}\n\n"
                    f"Verified topic: {_info_topic}\nQuestion: {question}"
                )
                print(
                    f"[rag] TOPIC-KNOWLEDGE MODE → topic='{_info_topic}' "
                    f"evidence_chunks={len(_gen_docs)} context_len={len(context)}",
                    file=sys.stderr,
                )
                try:
                    response = _langfuse_track_generation(
                        "topic", system_prompt, user_message,
                        question=question,
                        sources=_safe_source_list(_gen_metas),
                        context_size=len(context),
                        context_chunks=len(sources_out),
                        prompt_client=prompt_client,
                        root_obs=root_obs,
                    )
                    answer = response.choices[0].message.content
                    if answer and answer.strip():
                        return {"answer": answer.strip(), "sources": sources_out}
                    return {"answer": _not_covered_msg(file_label), "sources": sources_out}
                except Exception as exc:
                    print(f"[rag] Groq error in topic-knowledge mode: {exc}", file=sys.stderr)
                    return {
                        "answer": (
                            "⚠️ The AI service is currently unavailable. "
                            "Please try again in a moment or rephrase your question."
                        ),
                        "sources": [],
                    }

            # ── DOCUMENT FACT PATH ────────────────────────────────────────────
            # The topic is in the document and the question asks for a specific
            # fact stored IN the document (person's name, amount, date, etc.).
            # Fall through to the normal RAG retrieval path below.
            print(
                f"[query] type=document_fact topic='{_info_topic}' → proceeding to RAG",
                file=sys.stderr,
            )

        else:
            # Topic NOT in document → block all general knowledge
            print(f"[topic] verified=False", file=sys.stderr)
            print(f"[topic] knowledge_allowed=False", file=sys.stderr)
            print(f"[rag] blocked unverified topic '{_info_topic}'", file=sys.stderr)
            if root_obs is not None:
                try:
                    root_obs.update(
                        metadata={
                            "mode": "document_qa",
                            "topic_gate": _info_topic,
                            "topic_verified": False,
                            "status": "success",
                        },
                        output={
                            "answer": _not_covered_msg(file_label),
                            "sources": [],
                            "mode": "document_qa",
                            "status": "success",
                        },
                    )
                except Exception:
                    pass
            return {"answer": _not_covered_msg(file_label), "sources": []}


    # ── NORMAL DOCUMENT-GROUNDED RAG MODE ────────────────────────
    def _run_rag_retrieval():
        stats = {}
        docs, distances, metas = _retrieve_relevant_chunks(
            collection, question, doc_count, active_source, target_source=target_source, out_stats=stats
        )
        return (docs, distances, metas), stats

    (docs, distances, metas), ret_lat_ms = _langfuse_track_retrieval(
        question, sources_found, target_source, doc_count,
        _run_rag_retrieval,
        root_obs=root_obs,
        source_func=_retrieve_relevant_chunks,
    )

    # FINAL LEXICAL FALLBACK FOR DOCUMENT FACTS
    if not docs:
        fallback_keywords = [
            kw for kw in _extract_question_keywords(question)
            if len(kw) > 2 and kw not in {
                "candidate", "information", "document", "tell", "tells",
                "telling", "told", "details", "what", "where", "when", "about",
            }
        ]
        try:
            all_data = collection.get(include=["documents", "metadatas"])
            all_docs = all_data.get("documents", []) or []
            all_metas = all_data.get("metadatas", []) or []
            fallback_docs, fallback_metas = [], []
            patterns = [
                re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.IGNORECASE)
                for kw in fallback_keywords
            ]
            for doc, meta in zip(all_docs, all_metas or [{}] * len(all_docs)):
                if target_source and (meta or {}).get("source") != target_source:
                    continue
                norm_doc = _normalise_for_term_search(doc or "")
                if any(pattern.search(norm_doc) for pattern in patterns):
                    fallback_docs.append(doc)
                    fallback_metas.append(meta or {})
            if fallback_docs:
                docs = fallback_docs[:RERANK_TOP_N]
                metas = fallback_metas[:RERANK_TOP_N]
                distances = [0.01] * len(docs)
                print(
                    f"[rag] FACT LEXICAL FALLBACK: found {len(fallback_docs)} matching chunks "
                    f"for keywords={fallback_keywords}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[rag] fact lexical fallback failed: {exc}", file=sys.stderr)

    if not docs:
        if root_obs is not None:
            root_obs.update(
                metadata={
                    "mode": mode,
                    "retrieved_sources": [],
                    "retrieval_latency_ms": ret_lat_ms,
                    "status": "success",
                },
                output={
                    "answer": _not_covered_msg(file_label),
                    "sources": [],
                    "mode": mode,
                    "status": "success",
                },
            )
        return {"answer": _not_covered_msg(file_label), "sources": []}

    # A referenced document section may span multiple OCR chunks/pages. Give
    # those section-level requests a larger evidence window; ordinary queries
    # keep the existing smaller context budget.
    structured_refs = _extract_structured_references(question)
    context_budget = 10000 if structured_refs else MAX_CONTEXT_CHARS
    context, sources_out = _build_bounded_context(
        docs, metas, active_source, max_chars=context_budget
    )

    if not context.strip():
        if root_obs is not None:
            root_obs.update(
                metadata={
                    "mode": mode,
                    "retrieved_sources": _safe_source_list(metas),
                    "retrieval_latency_ms": ret_lat_ms,
                    "status": "success",
                },
                output={
                    "answer": _not_covered_msg(file_label),
                    "sources": sources_out,
                    "mode": mode,
                    "status": "success",
                },
            )
        return {"answer": _not_covered_msg(file_label), "sources": sources_out}

    if _groq_client is None:
        return {
            "answer": (
                "⚠️ The AI backend is not configured. "
                "Please set GROQ_API_KEY in backend/.env and restart the server."
            ),
            "sources": [],
        }

    system_prompt, prompt_client = _get_managed_prompt(
        "rag/querymind-document-qa",
        file_label=file_label,
    )
    user_message = f"Document context:\n{context}\n\nQuestion: {question}"

    print(
        f"[rag] → Groq: {len(sources_out)} evidence chunks, "
        f"{len(context)} chars, max_tokens={MAX_TOKENS}, temperature=0",
        file=sys.stderr,
    )

    try:
        response = _langfuse_track_generation(
            mode, system_prompt, user_message,
            question=question,
            sources=_safe_source_list(metas),
            context_size=len(context),
            context_chunks=len(sources_out),
                prompt_client=prompt_client,
            root_obs=root_obs,
        )

        answer = response.choices[0].message.content
        if answer and answer.strip():
            print(f"[rag] ANSWER: {answer.strip()[:200]}", file=sys.stderr)
            return {"answer": answer.strip(), "sources": sources_out}

        return {"answer": _not_covered_msg(file_label), "sources": sources_out}

    except Exception as exc:
        print(f"[rag] Groq error in RAG mode: {exc}", file=sys.stderr)
        return {
            "answer": (
                "⚠️ The AI service is currently unavailable. "
                "Please try again in a moment or rephrase your question."
            ),
            "sources": [],
        }

def answer_question(
    chat_id: str,
    question: str,
    selected_source: Optional[str] = None,
    user_email: Optional[str] = None,
    conversation_history: Optional[list[dict]] = None,
) -> dict:
    """Public answer API with Langfuse tracing around the existing RAG pipeline."""
    if not _langfuse_enabled():
        return _answer_question_impl(
            chat_id, question, selected_source=selected_source, user_id=None, conversation_history=conversation_history, user_email=user_email
        )

    t_start = time.perf_counter()
    user_id = _anonymize_user(user_email, chat_id)
    is_def = _extract_definition_term(question) is not None
    is_vis = _is_visual_intent(question)
    initial_mode = "definition" if is_def else ("image_qa" if is_vis else "document_qa")

    try:
        prop_mgr = (
            propagate_attributes(
                user_id=user_id,
                session_id=str(chat_id),
                trace_name="querymind-chat",
                metadata={
                    "chat_id": str(chat_id),
                    "selected_source": str(selected_source or ""),
                    "mode": initial_mode,
                },
            )
            if propagate_attributes is not None
            else contextlib.nullcontext()
        )

        with prop_mgr:
            with _langfuse.start_as_current_observation(
                as_type="chain",
                name="querymind-chat",
                input={
                    "question": question,
                    "chat_id": chat_id,
                    "selected_source": selected_source or "",
                    "user_id": user_id,
                },
            ) as root_obs:
                try:
                    result = _answer_question_impl(
                        chat_id,
                        question,
                        selected_source=selected_source,
                        user_id=user_id,
                        root_obs=root_obs,
                        conversation_history=conversation_history,
                        user_email=user_email,
                    )
                    t_end = time.perf_counter()
                    total_latency_ms = round((t_end - t_start) * 1000, 2)
                    answer_str = result.get("answer", "") if isinstance(result, dict) else str(result)
                    sources_list = [
                        s.get("label", "")
                        for s in (result.get("sources", []) if isinstance(result, dict) else [])
                        if isinstance(s, dict)
                    ]
                    is_err = "⚠️" in answer_str and ("unavailable" in answer_str or "unexpected error" in answer_str)
                    final_status = "error" if is_err else "success"

                    root_obs.update(
                        output={
                            "answer": answer_str[:4000],
                            "sources": sources_list,
                            "status": final_status,
                            "total_latency_ms": total_latency_ms,
                        },
                        metadata={
                            "status": final_status,
                            "total_latency_ms": total_latency_ms,
                            "retrieved_sources": sources_list,
                        },
                        level="ERROR" if is_err else "DEFAULT",
                        status_message=answer_str[:300] if is_err else None,
                    )
                    return result
                except Exception as exc:
                    t_end = time.perf_counter()
                    total_latency_ms = round((t_end - t_start) * 1000, 2)
                    root_obs.update(
                        output={"error": str(exc)[:500], "status": "error"},
                        level="ERROR",
                        status_message=str(exc)[:400],
                        metadata={"status": "error", "total_latency_ms": total_latency_ms},
                    )
                    raise
    finally:
        try:
            _langfuse.flush()
        except Exception:
            pass
