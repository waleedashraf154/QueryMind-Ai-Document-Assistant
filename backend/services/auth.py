"""
backend/services/auth.py
────────────────────────
Face embedding persistence backed by Supabase.

Functions:
- save_face_embedding(email, embedding)     -> tuple[bool, str]
- delete_face_embedding(email)              -> tuple[bool, str]
- get_face_embeddings_for_matching()        -> list[dict]
- get_user_by_email(email)                  -> dict | None
- has_face_registered(email)                -> bool
"""

from typing import Optional, Tuple, Dict, Any, List
from backend.database.supabase import get_supabase_client


def _ensure_user_row(client: Any, clean_email: str) -> None:
    """
    Ensure a row exists in `users` for the given email so the FK constraint
    on `face_embeddings.email` is satisfied.
    Clerk users won't have a password_hash — we insert a placeholder.
    """
    try:
        res = client.table("users").select("email").eq("email", clean_email).execute()
        if not res.data or len(res.data) == 0:
            username = clean_email.split("@")[0]
            client.table("users").insert({
                "email": clean_email,
                "username": username,
                "password_hash": "external_user",   # Clerk-managed; no local password
            }).execute()
    except Exception as exc:
        # Non-fatal: if the insert fails (e.g., race condition), proceed anyway
        print(f"[_ensure_user_row] Could not auto-create user row: {exc}")


def save_face_embedding(email: str, embedding: Any) -> Tuple[bool, str]:
    """
    Insert or update face embedding in Supabase for the given user email.
    Converts numpy arrays or lists to JSON-serializable list of floats.
    Auto-creates a stub users row for Clerk-authenticated accounts that have
    no legacy password record (satisfies the face_embeddings FK constraint).
    """
    client = get_supabase_client()
    clean_email = email.strip().lower()

    # Convert to list of floats
    if hasattr(embedding, "flatten"):
        emb_list = [float(x) for x in embedding.flatten().tolist()]
    elif isinstance(embedding, (list, tuple)):
        emb_list = [float(x) for x in embedding]
    else:
        return False, "Invalid embedding format."

    try:
        # Guarantee the users row exists (Clerk users may not have one)
        _ensure_user_row(client, clean_email)

        # Replace any existing embedding
        existing = client.table("face_embeddings").select("id").eq("email", clean_email).execute()
        if existing.data and len(existing.data) > 0:
            client.table("face_embeddings").delete().eq("email", clean_email).execute()

        client.table("face_embeddings").insert({
            "email": clean_email,
            "embedding": emb_list,
        }).execute()

        return True, "Face ID registered successfully."

    except Exception as exc:
        return False, f"Failed to save Face ID: {exc}"


def delete_face_embedding(email: str) -> Tuple[bool, str]:
    """
    Delete the face embedding for the given user email from Supabase.
    Returns (True, "success") or (False, "error reason").
    """
    client = get_supabase_client()
    clean_email = email.strip().lower()
    try:
        existing = client.table("face_embeddings").select("id").eq("email", clean_email).execute()
        if not existing.data or len(existing.data) == 0:
            return False, "No Face ID is registered for this account."
        client.table("face_embeddings").delete().eq("email", clean_email).execute()
        return True, "Face ID deleted successfully."
    except Exception as exc:
        return False, f"Failed to delete Face ID: {exc}"


def get_face_embeddings_for_matching() -> List[Dict[str, Any]]:
    """
    Fetch all (email, embedding) pairs from Supabase for face similarity matching.
    Returns list of dicts: [{'email': '...', 'embedding': [float, ...]}, ...]
    """
    client = get_supabase_client()
    try:
        res = client.table("face_embeddings").select("email, embedding").execute()
        return res.data or []
    except Exception as exc:
        print(f"Error fetching face embeddings: {exc}")
        return []


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch user record by email."""
    client = get_supabase_client()
    clean_email = email.strip().lower()
    try:
        res = client.table("users").select("email, username").eq("email", clean_email).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception:
        return None


def has_face_registered(email: str) -> bool:
    """Check if the user has a registered Face ID in Supabase."""
    client = get_supabase_client()
    clean_email = email.strip().lower()
    try:
        res = client.table("face_embeddings").select("id").eq("email", clean_email).execute()
        return bool(res.data and len(res.data) > 0)
    except Exception:
        return False
