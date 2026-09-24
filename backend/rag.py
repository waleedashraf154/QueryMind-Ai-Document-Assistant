"""Compatibility facade for QueryMind's RAG package.

Implementation is organized under :mod:`backend.rag_pipeline`; this module
keeps the existing `backend.rag` import path stable.
"""
from .rag_pipeline import answer_question
from .rag_pipeline.vector_store import add_document, get_chroma_collection
from .rag_pipeline.chunking import chunk_text
from .rag_pipeline.embeddings import embed_texts

__all__ = ["answer_question", "add_document", "get_chroma_collection", "chunk_text", "embed_texts"]
