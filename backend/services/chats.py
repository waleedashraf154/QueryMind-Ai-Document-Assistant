"""Chat management application services."""
import shutil
import sys
from pathlib import Path

import backend.database.storage as storage
from backend.core.config import STORAGE_DIR
from backend.rag import get_chroma_collection

def create_chat(user_email: str, chat_id: str, title: str) -> None:
    storage.create_chat(user_email, chat_id, title or "New Chat")

def get_chats(user_email: str):
    return storage.get_chats_for_user(user_email)

def get_sources(chat_id: str, user_email: str | None = None) -> dict:
    if user_email:
        storage.require_chat_owner(chat_id, user_email.strip().lower())

    sources = []
    seen = set()
    try:
        col = get_chroma_collection(chat_id)
        if col.count() > 0:
            data = col.get(limit=100, include=["metadatas", "documents"])
            metas = data.get("metadatas", []) or []
            docs = data.get("documents", []) or []
            for meta, doc in zip(metas, docs):
                src = (meta or {}).get("source", "")
                if src and src not in seen:
                    seen.add(src)
                    sources.append({"label": src, "snippet": (doc[:300] if doc else "")})
    except Exception as exc:
        print(f"Non-fatal error querying chat collection: {exc}")

    # 2. From local chat storage if not yet indexed
    # IMPORTANT: Do not call extract_text() here. During background ingestion,
    # the document may still be undergoing OCR. Parsing it again from /sources
    # would start a second OCR job and can make this endpoint take several minutes.
    chat_dir = STORAGE_DIR / chat_id
    if chat_dir.exists() and chat_dir.is_dir():
        for p in chat_dir.iterdir():
            if p.is_file() and not p.name.startswith(".") and p.name not in seen:
                seen.add(p.name)
                sources.append({
                    "label": p.name,
                    "snippet": "Document attached to this chat. Processing may still be in progress.",
                })

    return {"sources": sources}

def delete_chat(chat_id: str, user_email: str) -> dict:
    success = storage.delete_chat(chat_id, user_email)
    chat_dir = STORAGE_DIR / chat_id
    if chat_dir.exists():
        shutil.rmtree(chat_dir, ignore_errors=True)
    if not success:
        raise RuntimeError("Failed to delete chat")
    return {"status": "ok", "chat_id": chat_id}

def rename_chat(chat_id: str, user_email: str, title: str) -> dict:
    clean_email = user_email.strip().lower()
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Chat title cannot be empty")
    storage.require_chat_owner(chat_id, clean_email)
    storage.rename_chat(chat_id, clean_title)
    return {"status": "ok", "chat_id": chat_id, "title": clean_title}
