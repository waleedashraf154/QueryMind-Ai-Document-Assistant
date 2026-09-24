"""Document application services."""
import hashlib
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import backend.database.storage as storage
from backend.core.config import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_SIZE_MB,
    STORAGE_DIR,
)
from backend.parsers.document_parser import extract_text
from backend.rag import add_document, get_chroma_collection

# ─────────────────────────────────────────────────────────────────────────────
# Async ingestion infrastructure
# ─────────────────────────────────────────────────────────────────────────────

# Bounded thread pool: max 2 concurrent ingestion tasks to avoid bursting
# Groq/Gemini API rate limits during multi-page OCR.
_ingestion_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")

# Registry: file_id -> { status, progress_message, filename, chat_id, hash, error }
# Lives in-process memory; reset on server restart (acceptable for development).
_ingestion_registry: dict[str, dict] = {}
_registry_lock = threading.Lock()

# SHA-256 deduplication index: "sha256:<hex>" -> file_id of already-indexed file
# Prevents re-OCR and re-embedding of identical file bytes.
_content_hash_index: dict[str, str] = {}
_hash_lock = threading.Lock()


class UploadTooLargeError(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_status(file_id: str, status: str, progress: str = "", error: str = "") -> None:
    with _registry_lock:
        entry = _ingestion_registry.get(file_id, {})
        entry["status"] = status
        entry["progress_message"] = progress
        if error:
            entry["error"] = error
        _ingestion_registry[file_id] = entry


def get_ingestion_status(file_id: str) -> dict:
    """Return the current ingestion status for a file_id."""
    with _registry_lock:
        entry = _ingestion_registry.get(file_id)
    if entry is None:
        return {
            "file_id": file_id,
            "filename": "",
            "status": "unknown",
            "progress_message": "File ID not found.",
            "error": "",
        }
    return {
        "file_id": file_id,
        "filename": entry.get("filename", ""),
        "status": entry.get("status", "unknown"),
        "progress_message": entry.get("progress_message", ""),
        "error": entry.get("error", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Fast save (returns immediately, ~50-200ms)
# ─────────────────────────────────────────────────────────────────────────────

async def save_document_fast(
    chat_id: str,
    user_email: str,
    file,
) -> dict:
    """
    Read the uploaded file bytes, save to local disk, queue the Supabase
    upload as a fire-and-forget background thread, and return a file_id
    immediately — without running OCR, embedding, or Chroma indexing.

    Returns
    -------
    dict
        { file_id, filename, status: "queued" }
    """
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: "
            f"{', '.join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))}"
        )

    clean_email = user_email.strip().lower()
    storage.require_chat_owner(chat_id, clean_email)

    chat_storage_dir = STORAGE_DIR / chat_id
    chat_storage_dir.mkdir(parents=True, exist_ok=True)
    save_path = chat_storage_dir / filename

    t0 = time.perf_counter()
    print(
        f"[PERF][UPLOAD] start - filename='{filename}' chat_id={chat_id}",
        file=sys.stderr,
    )

    # ── Read file bytes ────────────────────────────────────────────────────
    total_bytes = 0
    file_parts: list[bytes] = []
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise UploadTooLargeError(
                    f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit."
                )
            file_parts.append(chunk)
        file_bytes = b"".join(file_parts)
        save_path.write_bytes(file_bytes)
    except UploadTooLargeError:
        save_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to save file: {exc}") from exc

    dt_save = time.perf_counter() - t0
    print(
        f"[PERF][UPLOAD] file saved (duration={dt_save:.3f}s) filename='{filename}' size={total_bytes}B",
        file=sys.stderr,
    )

    # ── Assign stable file_id ──────────────────────────────────────────────
    file_id = str(uuid.uuid4())

    # ── Register status ────────────────────────────────────────────────────
    with _registry_lock:
        _ingestion_registry[file_id] = {
            "filename": filename,
            "chat_id": chat_id,
            "user_email": clean_email,
            "save_path": str(save_path),
            "file_size": total_bytes,
            "content_type": file.content_type,
            "file_bytes_ref": file_bytes,     # held until ingestion starts
            "status": "queued",
            "progress_message": "Queued for processing…",
            "error": "",
            "hash": "",
        }

    # ── Fire-and-forget Supabase upload (non-blocking) ─────────────────────
    def _supabase_upload():
        t_s = time.perf_counter()
        print(
            f"[PERF][UPLOAD] storage_upload start - filename='{filename}'",
            file=sys.stderr,
        )
        try:
            metadata = storage.save_user_document(
                clean_email, filename, file_bytes, total_bytes, file.content_type
            )
            dt_s = time.perf_counter() - t_s
            print(
                f"[PERF][UPLOAD] storage_upload complete (duration={dt_s:.3f}s) "
                f"path='{metadata.get('storage_path', '')}'",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[PERF][UPLOAD] storage_upload FAILED for '{filename}': {exc}",
                file=sys.stderr,
            )

    threading.Thread(target=_supabase_upload, daemon=True, name=f"supabase-{file_id}").start()

    # ── Submit background ingestion to bounded pool ────────────────────────
    _ingestion_pool.submit(_ingest_document_background, file_id)

    return {"file_id": file_id, "filename": filename, "status": "queued"}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Background ingestion (OCR → chunk → embed → index)
# ─────────────────────────────────────────────────────────────────────────────

def _ingest_document_background(file_id: str) -> None:
    """
    Run OCR, chunking, embedding, and Chroma indexing in a background thread.
    Updates _ingestion_registry throughout with status and PERF log lines.
    """
    with _registry_lock:
        entry = dict(_ingestion_registry.get(file_id, {}))

    if not entry:
        print(f"[ingest] file_id={file_id} not found in registry; skipping.", file=sys.stderr)
        return

    filename = entry["filename"]
    chat_id = entry["chat_id"]
    clean_email = entry["user_email"]
    save_path = Path(entry["save_path"])
    file_bytes: bytes = entry.get("file_bytes_ref", b"")

    t_total = time.perf_counter()
    _set_status(file_id, "processing", "Computing file fingerprint…")

    # ── SHA-256 deduplication ──────────────────────────────────────────────
    file_hash = "sha256:" + hashlib.sha256(file_bytes).hexdigest()
    with _registry_lock:
        _ingestion_registry[file_id]["hash"] = file_hash
        # Clear the in-memory bytes reference to free RAM now that we have the hash
        _ingestion_registry[file_id]["file_bytes_ref"] = None

    print(
        f"[ingest] '{filename}' hash={file_hash[:22]}... checking dedup cache",
        file=sys.stderr,
    )

    with _hash_lock:
        existing_fid = _content_hash_index.get(file_hash)

    if existing_fid and existing_fid != file_id:
        # ── Fast dedup path: copy vectors from the already-indexed file ───
        with _registry_lock:
            existing_entry = _ingestion_registry.get(existing_fid, {})
        existing_chat_id = existing_entry.get("chat_id", "")
        existing_filename = existing_entry.get("filename", filename)

        print(
            f"[ingest] DEDUP HIT: '{filename}' matches file_id={existing_fid} "
            f"(chat={existing_chat_id}); copying vectors → no OCR/embedding.",
            file=sys.stderr,
        )
        _set_status(file_id, "processing", "Reusing existing index (duplicate detected)…")

        copied = False
        try:
            if existing_chat_id:
                src_col = get_chroma_collection(existing_chat_id)
                cached = src_col.get(
                    where={"source": existing_filename},
                    include=["documents", "embeddings", "metadatas"],
                )
                docs = cached.get("documents") or []
                embs = cached.get("embeddings") or []
                metas = cached.get("metadatas") or []
                if docs and embs and len(docs) == len(embs):
                    if hasattr(embs, "tolist"):
                        embs = embs.tolist()
                    tgt_col = get_chroma_collection(chat_id)
                    safe_fn = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)
                    tgt_col.upsert(
                        ids=[f"{safe_fn}_{i}" for i in range(len(docs))],
                        embeddings=embs,
                        documents=docs,
                        metadatas=[{**m, "source": filename} for m in metas],
                    )
                    copied = True
                    print(
                        f"[ingest] Copied {len(docs)} chunks for '{filename}' from dedup cache.",
                        file=sys.stderr,
                    )
        except Exception as exc:
            print(
                f"[ingest] Non-fatal: dedup vector copy failed for '{filename}': {exc}; "
                "falling through to full extraction.",
                file=sys.stderr,
            )

        if copied:
            dt = time.perf_counter() - t_total
            print(
                f"[PERF][TOTAL] '{filename}' dedup path complete (duration={dt:.3f}s)",
                file=sys.stderr,
            )
            _set_status(file_id, "ready", "Ready")
            return

    # ── Full extraction path ───────────────────────────────────────────────
    _set_status(file_id, "processing", "Extracting text…")

    if not save_path.exists():
        _set_status(file_id, "failed", "", f"Saved file not found: {save_path}")
        print(f"[ingest] ERROR: save_path missing for '{filename}': {save_path}", file=sys.stderr)
        return

    try:
        t_ocr = time.perf_counter()
        print(f"[PERF][OCR] start - filename='{filename}'", file=sys.stderr)
        text = extract_text(str(save_path))
        dt_ocr = time.perf_counter() - t_ocr
        print(f"[PERF][OCR] end (duration={dt_ocr:.3f}s) chars={len(text)}", file=sys.stderr)
    except Exception as exc:
        _set_status(file_id, "failed", "", f"Text extraction failed: {exc}")
        print(f"[ingest] ERROR: extract_text failed for '{filename}': {exc}", file=sys.stderr)
        return

    if not text.strip():
        _set_status(file_id, "failed", "", f"No readable text found in '{filename}'.")
        print(f"[ingest] ERROR: no text extracted from '{filename}'.", file=sys.stderr)
        return

    _set_status(file_id, "processing", "Indexing document…")

    try:
        print(f"[PERF][CHUNK] start - filename='{filename}'", file=sys.stderr)
        # add_document internally handles chunk + embed + upsert with their own PERF logs
        add_document(chat_id, text, filename=filename)
        print(f"[PERF][CHUNK] end / [PERF][EMBED] end / [PERF][INDEX] end - filename='{filename}'", file=sys.stderr)
    except Exception as exc:
        _set_status(file_id, "failed", "", f"Indexing failed: {exc}")
        print(f"[ingest] ERROR: add_document failed for '{filename}': {exc}", file=sys.stderr)
        return

    # ── Register hash for future deduplication ─────────────────────────────
    with _hash_lock:
        _content_hash_index[file_hash] = file_id

    dt_total = time.perf_counter() - t_total
    print(
        f"[PERF][TOTAL] '{filename}' ingestion complete (duration={dt_total:.3f}s)",
        file=sys.stderr,
    )
    _set_status(file_id, "ready", "Ready")


# ─────────────────────────────────────────────────────────────────────────────
# Original synchronous upload (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def list_user_documents(user_email: str) -> list[dict]:
    """Return persistent user documents with legacy local fallback."""
    clean_email = user_email.strip().lower()
    if not clean_email:
        return []
    try:
        docs = storage.list_user_documents(clean_email)
        if docs is not None:
            return docs
    except Exception as exc:
        print(
            f"[documents] Supabase list failed; using legacy local fallback: {exc}",
            file=sys.stderr,
        )

    user_docs_dir = STORAGE_DIR / "user_documents" / clean_email
    if not user_docs_dir.exists():
        return []

    docs = []
    for p in user_docs_dir.iterdir():
        if p.is_file() and not p.name.startswith("."):
            try:
                stat = p.stat()
                docs.append({
                    "filename": p.name,
                    "size": stat.st_size,
                    "updated_at": stat.st_mtime,
                })
            except Exception:
                pass
    docs.sort(key=lambda d: d.get("updated_at", 0), reverse=True)
    return docs

def delete_documents(user_email: str, filenames: list[str]) -> list[str]:
    clean_email = user_email.strip().lower()
    if not clean_email:
        raise ValueError("User email is required.")
    deleted = storage.delete_user_documents(clean_email, filenames)
    for fn in filenames:
        safe_name = Path(fn).name
        legacy_file = STORAGE_DIR / "user_documents" / clean_email / safe_name
        if legacy_file.exists():
            try:
                legacy_file.unlink()
            except Exception:
                pass
    return deleted

MIME_MAP = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".txt": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

def resolve_mime_type(filename: str, declared_mime: Optional[str] = None) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "application/pdf"
    if declared_mime and declared_mime not in ("application/octet-stream", "binary/octet-stream"):
        return declared_mime
    return MIME_MAP.get(ext, "application/octet-stream")

def download_document(user_email: str, filename: str) -> tuple[bytes, dict, str]:
    clean_email = user_email.strip().lower()
    safe_filename = Path(filename).name
    try:
        file_bytes, meta = storage.download_user_document(clean_email, safe_filename)
        mime_type = resolve_mime_type(safe_filename, meta.get("mime_type"))
        return file_bytes, meta, mime_type
    except Exception:
        local_path = STORAGE_DIR / "user_documents" / clean_email / safe_filename
        if local_path.exists():
            mime_type = resolve_mime_type(safe_filename)
            return local_path.read_bytes(), {}, mime_type
        for match in STORAGE_DIR.glob(f"*/{safe_filename}"):
            if match.is_file():
                mime_type = resolve_mime_type(safe_filename)
                return match.read_bytes(), {}, mime_type
        raise

def preview_document(user_email: str, filename: str) -> dict:
    clean_email = user_email.strip().lower()
    safe_filename = Path(filename).name
    file_bytes, meta = storage.download_user_document(clean_email, safe_filename)
    ext = Path(safe_filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return {
        "filename": safe_filename,
        "text": text,
        "size": meta.get("file_size", len(file_bytes)),
        "mime_type": meta.get("mime_type"),
    }

def recall_document(chat_id: str, user_email: str, filename: str) -> dict:
    clean_email = user_email.strip().lower()
    safe_filename = Path(filename).name
    storage.require_chat_owner(chat_id, clean_email)

    chat_storage_dir = STORAGE_DIR / chat_id
    chat_storage_dir.mkdir(parents=True, exist_ok=True)
    save_path = chat_storage_dir / safe_filename
    target_col = get_chroma_collection(chat_id)

    try:
        file_bytes, metadata = storage.download_user_document(clean_email, safe_filename)
        save_path.write_bytes(file_bytes)
        print(
            f"[documents] Downloaded '{safe_filename}' from Supabase "
            f"path='{metadata['storage_path']}' for user={clean_email}",
            file=sys.stderr,
        )
    except FileNotFoundError:
        legacy_path = STORAGE_DIR / "user_documents" / clean_email / safe_filename
        if not legacy_path.exists():
            raise
        shutil.copy2(legacy_path, save_path)

    try:
        existing = target_col.get(where={"source": safe_filename}, limit=1)
        if existing and existing.get("ids"):
            return {"status": "ok", "filename": safe_filename, "already_indexed": True}
    except Exception as exc:
        print(f"[documents] Existing-index check failed: {exc}", file=sys.stderr)

    copied_vectors = False
    try:
        user_chats = storage.get_chats_for_user(clean_email)
        for other_cid in user_chats.keys():
            if other_cid == chat_id:
                continue
            other_col = get_chroma_collection(other_cid)
            cached = other_col.get(
                where={"source": safe_filename},
                include=["documents", "embeddings", "metadatas"],
            )
            docs = cached.get("documents")
            embs = cached.get("embeddings")
            metas = cached.get("metadatas") or []
            if docs and embs and len(docs) == len(embs):
                if hasattr(embs, "tolist"):
                    embs = embs.tolist()
                safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", safe_filename)
                target_col.upsert(
                    ids=[f"{safe_id}_{i}" for i in range(len(docs))],
                    embeddings=embs,
                    documents=docs,
                    metadatas=metas,
                )
                copied_vectors = True
                break
    except Exception as exc:
        print(f"[documents] Non-fatal vector reuse attempt failed: {exc}", file=sys.stderr)

    if not copied_vectors:
        text = extract_text(str(save_path))
        if not text.strip():
            raise ValueError(f"No readable text found in '{safe_filename}'.")
        add_document(chat_id, text, filename=safe_filename)

    return {"status": "ok", "filename": safe_filename, "already_indexed": False}

async def upload_document(chat_id: str, user_email: str, file) -> dict:
    """Original synchronous upload — kept for backward compatibility."""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: "
            f"{', '.join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))}"
        )

    clean_email = user_email.strip().lower()
    storage.require_chat_owner(chat_id, clean_email)

    chat_storage_dir = STORAGE_DIR / chat_id
    chat_storage_dir.mkdir(parents=True, exist_ok=True)
    save_path = chat_storage_dir / filename

    total_bytes = 0
    file_parts: list[bytes] = []
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise UploadTooLargeError(
                    f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit."
                )
            file_parts.append(chunk)
        file_bytes = b"".join(file_parts)
        save_path.write_bytes(file_bytes)
    except UploadTooLargeError:
        save_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to save file: {exc}") from exc

    t_upload_start = time.perf_counter()
    print(
        f"[PERF] upload total start - filename='{filename}' size={total_bytes}B",
        file=sys.stderr,
    )

    try:
        text = extract_text(str(save_path))
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise ValueError(f"Could not extract text from '{filename}': {exc}") from exc

    if not text.strip():
        save_path.unlink(missing_ok=True)
        raise ValueError(f"No readable text found in '{filename}'.")

    try:
        metadata = storage.save_user_document(
            clean_email, filename, file_bytes, total_bytes, file.content_type
        )
        print(
            f"[documents] Uploaded '{filename}' to Supabase "
            f"path='{metadata['storage_path']}' for user={clean_email}",
            file=sys.stderr,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to store '{filename}' in the persistent document library: {exc}"
        ) from exc

    try:
        add_document(chat_id, text, filename=filename)
    except Exception as exc:
        raise RuntimeError(f"Failed to index document: {exc}") from exc

    t_upload_end = time.perf_counter()
    print(
        f"[PERF] upload total end (duration={t_upload_end - t_upload_start:.3f}s)",
        file=sys.stderr,
    )

    return {"status": "ok", "filename": filename, "char_count": len(text)}
