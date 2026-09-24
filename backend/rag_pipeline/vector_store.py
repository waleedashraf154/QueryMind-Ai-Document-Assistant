from __future__ import annotations
import os
import re
import sys
import uuid
import time

from .common import _chroma_client, _EMBED_DIM
from .chunking import chunk_text
from .document_structure import split_numbered_sections
from .embeddings import embed_texts

def get_chroma_collection(chat_id: str):
    """
    Return (or create) a persistent ChromaDB collection for *chat_id*.

    Each chat gets its own isolated collection so documents from different
    chats never mix.
    """
    if _chroma_client is None:
        raise RuntimeError("chromadb is not installed.")

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", chat_id)[:63]
    if len(safe_name) < 3:
        safe_name = safe_name.ljust(3, "0")

    collection = _chroma_client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Dimension-mismatch guard ─────────────────────────────────
    # Collections created before the switch to local sentence-transformers
    # embeddings (384-dim) may still hold old Gemini-embedding-001 vectors
    # (3072-dim, or whatever Gemini's model produced). Mixing dimensions in
    # the same HNSW index either errors out or silently corrupts distance
    # calculations. If a stale collection is detected, wipe and recreate it
    # so it gets cleanly re-populated on next upload.
    try:
        if collection.count() > 0:
            sample = collection.peek(limit=1)
            sample_embeddings = sample.get("embeddings")
            if sample_embeddings is not None and len(sample_embeddings) > 0:
                existing_dim = len(sample_embeddings[0])
                if existing_dim != _EMBED_DIM:
                    print(
                        f"[rag] Stale collection '{safe_name}' has {existing_dim}-dim "
                        f"embeddings, expected {_EMBED_DIM}. Resetting collection — "
                        f"documents in this chat must be re-uploaded.",
                        file=sys.stderr,
                    )
                    _chroma_client.delete_collection(safe_name)
                    collection = _chroma_client.get_or_create_collection(
                        name=safe_name,
                        metadata={"hnsw:space": "cosine"},
                    )
    except Exception as exc:
        print(f"[rag] Non-fatal dimension-check error for '{safe_name}': {exc}", file=sys.stderr)

    return collection

def add_document(chat_id: str, text: str, filename: str = "") -> None:
    """
    Chunk *text*, embed the chunks, and upsert them into the chat's
    ChromaDB collection with file type and filename metadata.
    """
    if filename:
        filename = os.path.basename(filename.strip())
    is_img = any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"])
    file_type = "image" if is_img else "document"

    chunks = chunk_text(text)
    if not chunks:
        return

    # Keep exact section structure alongside vectors. This is used for named-section
    # retrieval and prevents a fixed top-k search from truncating an entire section.
    section_meta: list[dict] = [{} for _ in chunks]
    try:
        structured_chunks, structured_meta = split_numbered_sections(text)
        if structured_chunks and structured_chunks == chunks and len(structured_meta) == len(chunks):
            section_meta = structured_meta
    except Exception as exc:
        print(f"[rag] Non-fatal section metadata extraction error: {exc}", file=sys.stderr)

    collection = get_chroma_collection(chat_id)

    # Delete existing chunks belonging to this document to prevent stale/orphaned chunks
    if filename:
        try:
            collection.delete(where={"source": filename})
        except Exception as exc:
            print(
                f"[rag] Pre-upsert cleanup for '{filename}' in chat '{chat_id}': {exc}",
                file=sys.stderr,
            )

    if filename:
        safe_fn = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)
        ids = [f"{safe_fn}_{i}" for i in range(len(chunks))]
    else:
        ids = [str(uuid.uuid4()) for _ in chunks]

    t_emb_start = time.perf_counter()
    print(f"[PERF] embedding start - chunks={len(chunks)}", file=sys.stderr)
    embeddings = embed_texts(chunks)
    t_emb_end = time.perf_counter()
    print(f"[PERF] embedding end (duration={t_emb_end - t_emb_start:.3f}s)", file=sys.stderr)

    metadatas = []
    current_page = 1
    for chunk_idx, chunk in enumerate(chunks):
        page_match = re.search(r"\[OCR content from PDF page\s+(\d+)\]", chunk, re.IGNORECASE)
        if page_match:
            current_page = int(page_match.group(1))
            chunk_type = "pdf_ocr"
        elif re.search(r"(?:^|\n)---\s*page\s+(\d+)\b", chunk, re.IGNORECASE):
            current_page = int(re.search(r"(?:^|\n)---\s*page\s+(\d+)\b", chunk, re.IGNORECASE).group(1))
            chunk_type = "pdf_ocr" if "[OCR content" in chunk else file_type
        else:
            chunk_type = "pdf_ocr" if "[OCR content" in chunk else file_type

        meta = {
            "source": filename,
            "type": chunk_type,
            "page": current_page,
            "chunk_index": chunk_idx,
        }
        if chunk_idx < len(section_meta):
            sm = section_meta[chunk_idx] or {}
            if sm.get("section_number"):
                meta["section_number"] = str(sm.get("section_number"))
                meta["section_title"] = str(sm.get("section_title") or "")
                meta["section_key"] = str(sm.get("section_key") or "")
                meta["section_level"] = int(sm.get("section_level") or 1)

        # Resume/CV structure-aware chunks carry an explicit "Section: ..." prefix.
        # Persist that heading as metadata too, so named section retrieval works for
        # non-numbered documents as reliably as it does for numbered reports.
        first_line = (chunk or "").splitlines()[0].strip() if (chunk or "").splitlines() else ""
        m_resume_section = re.match(r"^Section:\s*(.+?)\s*$", first_line, re.IGNORECASE)
        if m_resume_section:
            resume_title = re.sub(r"\s+", " ", m_resume_section.group(1)).strip()
            if resume_title and resume_title.casefold() not in {"profile", "reference"}:
                meta["section_title"] = resume_title
                meta["section_key"] = re.sub(r"\s+", " ", resume_title.casefold()).strip()
                meta["section_level"] = 1
        metadatas.append(meta)

    t_idx_start = time.perf_counter()
    print(f"[PERF] indexing start - collection='{chat_id}' chunks={len(chunks)}", file=sys.stderr)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    t_idx_end = time.perf_counter()
    print(f"[PERF] indexing end (duration={t_idx_end - t_idx_start:.3f}s)", file=sys.stderr)
    print(
        f"[rag] Indexed {len(chunks)} chunks for '{filename}' into chat '{chat_id}'.",
        file=sys.stderr,
    )
