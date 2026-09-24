"""
Persistent chat history backed by Supabase.
"""
from typing import Dict, Any
from pathlib import Path
from backend.database.supabase import get_supabase_client, reset_supabase_client


DOCUMENTS_BUCKET = "documents"


def _clean_email(user_email: str) -> str:
    return user_email.strip().lower()


def _storage_path_for_user(user_email: str, filename: str) -> str:
    clean_email = _clean_email(user_email)
    return f"{clean_email}/{Path(filename).name}"


def save_user_document(
    user_email: str,
    filename: str,
    file_bytes: bytes,
    file_size: int,
    mime_type: str | None = None,
) -> dict:
    """Persist a user's original document in private Supabase Storage."""
    client = get_supabase_client()
    clean_email = _clean_email(user_email)
    safe_filename = Path(filename).name
    storage_path = _storage_path_for_user(clean_email, safe_filename)

    client.storage.from_(DOCUMENTS_BUCKET).upload(
        storage_path,
        file_bytes,
        file_options={
            "content-type": mime_type or "application/octet-stream",
            "upsert": "true",
        },
    )

    metadata = {
        "user_id": clean_email,
        "filename": safe_filename,
        "storage_path": storage_path,
        "file_size": int(file_size),
        "mime_type": mime_type,
    }

    existing = (
        client.table("documents")
        .select("id")
        .eq("user_id", clean_email)
        .eq("filename", safe_filename)
        .limit(1)
        .execute()
    )

    if existing.data:
        client.table("documents").update(metadata).eq(
            "id", existing.data[0]["id"]
        ).execute()
    else:
        client.table("documents").insert(metadata).execute()

    return metadata


def list_user_documents(user_email: str) -> list[dict]:
    """List persistent documents for a user from Supabase metadata."""
    client = get_supabase_client()
    clean_email = _clean_email(user_email)
    if not clean_email:
        return []

    result = (
        client.table("documents")
        .select("id, filename, storage_path, file_size, mime_type, created_at")
        .eq("user_id", clean_email)
        .order("created_at", desc=True)
        .execute()
    )

    return [
        {
            "id": row.get("id"),
            "filename": row.get("filename", ""),
            "size": int(row.get("file_size") or 0),
            "mime_type": row.get("mime_type"),
            "storage_path": row.get("storage_path", ""),
            "updated_at": row.get("created_at"),
        }
        for row in (result.data or [])
    ]


def download_user_document(user_email: str, filename: str) -> tuple[bytes, dict]:
    """Download a private user document from Supabase Storage."""
    client = get_supabase_client()
    clean_email = _clean_email(user_email)
    safe_filename = Path(filename).name

    result = (
        client.table("documents")
        .select("id, filename, storage_path, file_size, mime_type, created_at")
        .eq("user_id", clean_email)
        .eq("filename", safe_filename)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise FileNotFoundError(
            f"Document '{safe_filename}' not found in the user's document library."
        )

    row = result.data[0]
    try:
        data = client.storage.from_(DOCUMENTS_BUCKET).download(row["storage_path"])
    except Exception:
        # If connection was terminated or dropped, reset client and retry once with fresh connection
        reset_supabase_client()
        fresh_client = get_supabase_client()
        data = fresh_client.storage.from_(DOCUMENTS_BUCKET).download(row["storage_path"])

    return bytes(data), {
        "id": row.get("id"),
        "filename": row.get("filename", safe_filename),
        "storage_path": row.get("storage_path", ""),
        "file_size": int(row.get("file_size") or len(data)),
        "mime_type": row.get("mime_type"),
        "created_at": row.get("created_at"),
    }



def create_chat(user_email: str, chat_id: str, title: str = "New Chat") -> None:
    client = get_supabase_client()
    clean_email = user_email.strip().lower()
    if not clean_email or not chat_id.strip():
        raise ValueError("User email and chat ID are required.")

    res = client.table("chats").select("chat_id, user_email").eq("chat_id", chat_id).execute()
    if res.data:
        owner = (res.data[0].get("user_email") or "").strip().lower()
        if owner != clean_email:
            raise PermissionError("You do not have access to this chat.")
        return

    user_res = client.table("users").select("email").eq("email", clean_email).execute()
    if not user_res.data:
        client.table("users").insert({
            "email": clean_email,
            "username": clean_email.split("@")[0],
            "password_hash": "external_user",
        }).execute()

    insert_data = {
        "chat_id": chat_id,
        "user_email": clean_email,
        "title": title or "New Chat",
        "title_generated": False,
    }
    try:
        client.table("chats").insert(insert_data).execute()
    except Exception as exc:
        if "title_generated" in str(exc):
            insert_data.pop("title_generated", None)
            client.table("chats").insert(insert_data).execute()
        else:
            raise


def chat_belongs_to_user(chat_id: str, user_email: str) -> bool:
    client = get_supabase_client()
    clean_email = user_email.strip().lower()
    try:
        res = client.table("chats").select("chat_id").eq(
            "chat_id", chat_id
        ).eq("user_email", clean_email).execute()
        return bool(res.data)
    except Exception:
        return False


def require_chat_owner(chat_id: str, user_email: str) -> None:
    client = get_supabase_client()
    clean_email = user_email.strip().lower()
    try:
        res = client.table("chats").select("chat_id, user_email").eq("chat_id", chat_id).execute()
        if res.data:
            owner = (res.data[0].get("user_email") or "").strip().lower()
            if owner and owner != clean_email:
                raise PermissionError("You do not have access to this chat.")
            return
        # If chat doesn't exist yet in Supabase, create it for this user
        create_chat(clean_email, chat_id, "New Chat")
    except PermissionError:
        raise
    except Exception as exc:
        print(f"require_chat_owner non-fatal error: {exc}")


def rename_chat(chat_id: str, title: str) -> None:
    """Manually rename a chat and mark title_generated = True to permanently protect from auto-renaming."""
    client = get_supabase_client()
    clean_title = title.strip()
    try:
        client.table("chats").update({
            "title": clean_title,
            "title_generated": True,
        }).eq("chat_id", chat_id).execute()
    except Exception as exc:
        if "title_generated" in str(exc):
            client.table("chats").update({"title": clean_title}).eq("chat_id", chat_id).execute()
        else:
            raise


def is_chat_title_generated(chat_id: str) -> bool:
    """
    Check if automatic title has already been generated or if manual rename permanently protected it.
    Returns True if title_generated is True or if chat already has a custom/non-default title.
    """
    client = get_supabase_client()
    try:
        res = client.table("chats").select("title, title_generated").eq("chat_id", chat_id).execute()
        if not res.data:
            return False
        row = res.data[0]
        if "title_generated" in row and row["title_generated"] is not None:
            return bool(row["title_generated"])
        title = (row.get("title") or "").strip()
        return title not in ("New Chat", "Untitled chat", "")
    except Exception as exc:
        if "title_generated" in str(exc):
            try:
                fallback_res = client.table("chats").select("title").eq("chat_id", chat_id).execute()
                if fallback_res.data:
                    title = (fallback_res.data[0].get("title") or "").strip()
                    return title not in ("New Chat", "Untitled chat", "")
            except Exception:
                pass
        return False


def update_chat_title_if_not_generated(chat_id: str, title: str) -> bool:
    """
    Atomically update chat title if and only if title_generated is still FALSE.
    Protects against race conditions and duplicates from parallel requests.
    Returns True if updated successfully, False if already generated or updated.
    """
    client = get_supabase_client()
    clean_title = title.strip()
    try:
        res = client.table("chats").update({
            "title": clean_title,
            "title_generated": True,
        }).eq("chat_id", chat_id).eq("title_generated", False).execute()
        return bool(res.data)
    except Exception as exc:
        if "title_generated" in str(exc):
            try:
                check_res = client.table("chats").select("title").eq("chat_id", chat_id).execute()
                if check_res.data:
                    current = (check_res.data[0].get("title") or "").strip()
                    if current in ("New Chat", "Untitled chat", ""):
                        up_res = client.table("chats").update({"title": clean_title}).eq("chat_id", chat_id).execute()
                        return bool(up_res.data)
            except Exception:
                pass
        return False


def get_first_user_message(chat_id: str) -> str | None:
    """Retrieve the very first user message for a chat to generate an accurate topic title."""
    client = get_supabase_client()
    try:
        res = (
            client.table("messages")
            .select("content")
            .eq("chat_id", chat_id)
            .eq("role", "user")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        if res.data and len(res.data) > 0:
            return res.data[0].get("content")
    except Exception as exc:
        print(f"[storage] get_first_user_message non-fatal error: {exc}", file=sys.stderr)
    return None


def get_recent_messages(chat_id: str, limit: int = 12) -> list[dict]:
    """Return recent chat messages in chronological order for follow-up understanding."""
    client = get_supabase_client()
    try:
        res = (
            client.table("messages")
            .select("role, content, created_at")
            .eq("chat_id", chat_id)
            .order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        rows = list(reversed(res.data or []))
        return [
            {"role": str(r.get("role", "")), "content": str(r.get("content", "") or "")}
            for r in rows
            if r.get("content")
        ]
    except Exception as exc:
        print(f"[storage] get_recent_messages non-fatal error: {exc}")
        return []


def add_message(chat_id: str, role: str, content: str) -> None:
    client = get_supabase_client()
    client.table("messages").insert({
        "chat_id": chat_id,
        "role": role,
        "content": content,
    }).execute()


def get_chats_for_user(user_email: str) -> Dict[str, Any]:
    client = get_supabase_client()
    clean_email = user_email.strip().lower()
    try:
        chats_res = client.table("chats").select(
            "chat_id, title, title_generated, created_at"
        ).eq("user_email", clean_email).order("created_at", desc=True).execute()
    except Exception:
        chats_res = client.table("chats").select(
            "chat_id, title, created_at"
        ).eq("user_email", clean_email).order("created_at", desc=True).execute()

    if not chats_res.data:
        return {}

    result: Dict[str, Any] = {}
    chat_ids = []
    for c in chats_res.data:
        cid = c["chat_id"]
        chat_ids.append(cid)
        title_val = (c.get("title") or "").strip()
        if "title_generated" in c and c["title_generated"] is not None:
            is_gen = bool(c["title_generated"])
        else:
            is_gen = title_val not in ("New Chat", "Untitled chat", "")
        result[cid] = {
            "title": c["title"],
            "title_generated": is_gen,
            "messages": [],
        }

    msgs_res = client.table("messages").select(
        "chat_id, role, content, created_at"
    ).in_("chat_id", chat_ids).order("created_at", desc=False).execute()

    for m in (msgs_res.data or []):
        cid = m["chat_id"]
        if cid in result:
            result[cid]["messages"].append({
                "role": m["role"],
                "content": m["content"],
            })
    return result


def delete_chat(chat_id: str, user_email: str | None = None) -> bool:
    client = get_supabase_client()
    if user_email is not None:
        clean_email = user_email.strip().lower()
        res = client.table("chats").delete().eq(
            "chat_id", chat_id
        ).eq("user_email", clean_email).execute()
    else:
        res = client.table("chats").delete().eq("chat_id", chat_id).execute()
    return bool(res.data is not None)


def delete_user_documents(user_email: str, filenames: list[str]) -> list[str]:
    """Permanently delete user documents from Supabase Storage and metadata table."""
    client = get_supabase_client()
    clean_email = _clean_email(user_email)
    if not clean_email or not filenames:
        return []

    deleted: list[str] = []
    for fn in filenames:
        safe_name = Path(fn).name
        storage_path = _storage_path_for_user(clean_email, safe_name)

        # 1. Delete from Supabase Storage
        try:
            client.storage.from_(DOCUMENTS_BUCKET).remove([storage_path])
        except Exception as exc:
            print(f"[storage] Failed to delete storage file {storage_path}: {exc}")

        # 2. Delete from metadata table
        try:
            client.table("documents").delete().eq("user_id", clean_email).eq("filename", safe_name).execute()
        except Exception as exc:
            print(f"[storage] Failed to delete metadata for {safe_name}: {exc}")

        deleted.append(safe_name)

    return deleted

