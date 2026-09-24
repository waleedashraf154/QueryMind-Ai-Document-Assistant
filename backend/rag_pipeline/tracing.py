from __future__ import annotations
import contextlib
import hashlib
import inspect
import time
from pathlib import Path
from typing import Optional

try:
    from langfuse import propagate_attributes
except Exception:
    propagate_attributes = None

from backend.core.langfuse import (
    get_langfuse_client,
    get_managed_prompt,
    langfuse_enabled,
)
from .common import _BACKEND_DIR
from .query import _safe_source_list

_langfuse = get_langfuse_client()
def _langfuse_enabled() -> bool:
    return langfuse_enabled()

_get_managed_prompt = get_managed_prompt

def _anonymize_user(user_identifier: Optional[str], chat_id: str) -> str:
    """Generate a consistent, irreversible pseudonymous ID without raw personal data."""
    raw = (user_identifier or "").strip().lower()
    if raw:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"user_{digest}"
    chat_digest = hashlib.sha256(str(chat_id).encode("utf-8")).hexdigest()[:12]
    return f"anon_{chat_digest}"

def _get_source_location(func) -> dict:
    """Return source file, function name, and line range for a Python function."""
    try:
        source_file = inspect.getsourcefile(func) or ""
        source_lines, line_start = inspect.getsourcelines(func)

        source_path = Path(source_file).resolve()
        backend_root = Path(_BACKEND_DIR).resolve()
        try:
            source_file_display = str(source_path.relative_to(backend_root.parent))
        except ValueError:
            source_file_display = str(source_path)

        line_end = line_start + len(source_lines) - 1

        return {
            "source_file": source_file_display.replace("\\", "/"),
            "function": func.__name__,
            "line_start": line_start,
            "line_end": line_end,
        }
    except Exception:
        return {
            "source_file": "unknown",
            "function": getattr(func, "__name__", "unknown"),
            "line_start": 0,
            "line_end": 0,
        }

def _langfuse_track_retrieval(
    question: str,
    selected_sources: list[str],
    target_source: Optional[str],
    doc_count: int,
    fn,
    root_obs=None,
    source_func=None,
):
    """Run retrieval inside a Langfuse retriever observation named 'retrieval' without logging full document text."""
    if not _langfuse_enabled():
        res, _ = fn()
        return res, 0.0

    t_start = time.perf_counter()
    source_info = _get_source_location(source_func or fn)
    with _langfuse.start_as_current_observation(
        as_type="retriever",
        name="retrieval",
        input={
            "question": question,
            "selected_sources": selected_sources,
            "target_source": target_source,
        },
        metadata={
            "selected_sources": selected_sources,
            "target_source": target_source,
            "document_chunks_available": doc_count,
            "is_document_specific": bool(target_source),
            **source_info,
        },
    ) as retrieval_obs:
        try:
            result, stats = fn()
            t_end = time.perf_counter()
            retrieval_latency_ms = round((t_end - t_start) * 1000, 2)
            docs, distances, metas = result[:3]

            candidate_count = stats.get("candidate_count", len(docs))
            filtered_count = stats.get("filtered_count", len(docs))
            dedup_count = stats.get("dedup_count", len(docs))
            reranked_count = stats.get("reranked_count", len(docs))
            final_chunk_count = len(docs)
            dist_list = [round(float(d), 4) for d in distances]
            retrieved_sources = _safe_source_list(metas)

            output_payload = {
                "selected_sources": selected_sources,
                "target_source": target_source,
                "candidate_count": candidate_count,
                "filtered_count": filtered_count,
                "dedup_count": dedup_count,
                "reranked_count": reranked_count,
                "final_chunk_count": final_chunk_count,
                "distances": dist_list,
                "sources": retrieved_sources,
            }
            if "term" in stats:
                output_payload["term"] = stats["term"]
            if "term_found" in stats:
                output_payload["term_found"] = stats["term_found"]

            retrieval_obs.update(
                output=output_payload,
                metadata={
                    "latency_ms": retrieval_latency_ms,
                    "retrieved_sources": retrieved_sources,
                    "is_document_specific": bool(target_source),
                },
            )
            return result, retrieval_latency_ms
        except Exception as exc:
            t_end = time.perf_counter()
            retrieval_latency_ms = round((t_end - t_start) * 1000, 2)
            retrieval_obs.update(
                output={"error": str(exc)[:500]},
                level="ERROR",
                status_message=str(exc)[:500],
                metadata={"latency_ms": retrieval_latency_ms},
            )
            if root_obs is not None:
                root_obs.update(
                    level="ERROR",
                    status_message=f"Retrieval error: {str(exc)[:400]}",
                )
            raise
