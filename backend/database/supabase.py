"""
backend/database/supabase.py
─────────────
Supabase client initializer.
Reads SUPABASE_URL and SUPABASE_SERVICE_KEY from backend/.env via python-dotenv.
"""

import os
from typing import Optional
import httpx
from supabase import create_client, Client, ClientOptions

from backend.core.config import BACKEND_DIR

_client: Optional[Client] = None


def reset_supabase_client() -> None:
    """Reset the cached Supabase client singleton to force reconnection on stale sockets."""
    global _client
    if _client is not None:
        try:
            # Safely close underlying session if possible
            if hasattr(_client, "storage") and hasattr(_client.storage, "session"):
                _client.storage.session.close()
        except Exception:
            pass
    _client = None


def get_supabase_client() -> Client:
    """
    Return an initialized Supabase Client using service_role credentials.
    Configured with HTTP/1.1 (http2=False) to prevent HTTP/2 ConnectionTerminated errors.
    Raises RuntimeError if SUPABASE_URL or SUPABASE_SERVICE_KEY is missing.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials missing. Ensure SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY are set in backend/.env"
        )

    # Use HTTP/1.1 to prevent HTTP/2 idle stream termination issues on downloads
    http_transport_client = httpx.Client(
        http2=False,
        timeout=httpx.Timeout(60.0, connect=15.0),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0),
    )
    options = ClientOptions(httpx_client=http_transport_client)

    _client = create_client(url, key, options=options)
    return _client

