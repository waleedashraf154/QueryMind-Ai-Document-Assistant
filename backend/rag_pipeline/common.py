"""Shared RAG configuration, clients, and lightweight helpers."""
import os
import re
import sys
from pathlib import Path
from typing import Optional
import uuid

from backend.core.config import BACKEND_DIR

_BACKEND_DIR = str(BACKEND_DIR)

try:
    from backend.parsers.document_parser import extract_text as _extract_source_text
except Exception:
    _extract_source_text = None

GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

RELEVANCE_THRESHOLD: float = 0.70
RELEVANCE_THRESHOLD_FALLBACK: float = 0.78
NEAR_DUPLICATE_JACCARD: float = 0.85
TOP_K: int = 16
RERANK_TOP_N: int = 8
MAX_TOKENS: int = 4096
MAX_CONTEXT_CHARS: int = 9000

_HEADING_NAMES = (
    r"EDUCATION|EXPERIENCE|PROJECTS|CERTIFICATIONS?|AWARDS?|"
    r"SUMMARY|PROFESSIONAL\s+SUMMARY|EXECUTIVE\s+SUMMARY|OBJECTIVE|CAREER\s+OBJECTIVE|"
    r"PROFILE|PERSONAL\s+PROFILE|ABOUT|ABOUT\s+ME|"
    r"CONTACT|CONTACT\s+(?:DETAILS|INFORMATION)|"
    r"LANGUAGES?|REFERENCES?|"
    r"WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT\s+HISTORY|"
    r"(?:TECHNICAL\s+|SOFT\s+|KEY\s+|CORE\s+)?SKILLS|EXPERTISE|"
    r"PERSONAL\s+INFORMATION|PERSONAL\s+DETAILS|INFO|"
    r"HOBBIES|INTERESTS|ACHIEVEMENTS?|PUBLICATIONS?|VOLUNTEERING|EXTRACURRICULAR|"
    r"QUALIFICATIONS?|ACADEMIC\s+BACKGROUND|DECLARATION"
)
_HEADING_RE = re.compile(
    rf"^\s*(?:#+\s*)?({_HEADING_NAMES})(?:[\s:]*$|[\s:]+[\-–—:]?\s*(.+)$)",
    re.IGNORECASE,
)
_SUBENTRY_RE = re.compile(
    r"^\s*("
    r"Matric|Matriculation|SSC|HSSC|Intermediate|Graduation|Post\s*Graduation|"
    r"Bachelor(?:\s+of\s+[\w\s]+)?|Master(?:\s+of\s+[\w\s]+)?|PhD|Doctorate|O[- ]Levels?|A[- ]Levels?|"
    r"High\s*School|Secondary\s*School|Higher\s*Secondary"
    r")(\s*[:\-\u2013\u2014]|\s*$)",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^[\s]*([•\-\*▪◦]|\d+[\.\)])\s+")

try:
    import chromadb  # type: ignore
    _CHROMA_DIR = os.path.join(_BACKEND_DIR, "chroma_db")
    os.makedirs(_CHROMA_DIR, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)
except ImportError:
    _chroma_client = None  # type: ignore

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBED_DIM = 384
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
except ImportError:
    SentenceTransformer = None  # type: ignore
    _embed_model = None  # type: ignore

try:
    from groq import Groq  # type: ignore
    _groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    Groq = None  # type: ignore
    _groq_client = None  # type: ignore

_NO_DOCUMENT_MSG = (
    "Please upload a document first — I can only answer questions about "
    "files you've uploaded."
)

BATCH_SIZE = 64

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "used",
    "ought", "and", "or", "but", "nor", "so", "yet", "for", "if", "in",
    "on", "at", "to", "of", "by", "from", "with", "about", "as", "into",
    "than", "that", "this", "these", "those", "it", "its", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "me", "my",
    "we", "our", "you", "your", "he", "she", "they", "their", "him",
    "her", "i",
})

_FIELD_ALIASES: dict[str, list[str]] = {
    "intermediate": ["intermediate", "ics", "phy"],
    "matric":       ["matric", "ssc", "matriculation"],
    "graduation":   ["graduation", "bachelor", "bbit", "bs", "degree", "university"],
    "experience":   ["experience", "worked", "company", "employer", "position", "role"],
    "role":         ["role", "position", "designation", "title", "job"],
    "position":     ["position", "role", "designation", "title", "job"],
    "skills":       ["skills", "skill", "technologies", "tools"],
    "name":         ["name", "candidate"],
    "email":        ["email", "contact", "phone", "address"],
    "phone":        ["phone", "contact", "mobile", "cell", "tel"],
    "contact":      ["contact", "email", "phone", "mobile", "address"],
    "degree":       ["degree", "bachelor", "master", "doctorate", "phd"],
    "nationality":  ["nationality", "citizen", "citizenship"],
    "birth":        ["birth", "dob", "born"],
    "dob":          ["dob", "birth", "born"],
    "languages":    ["languages", "language"],
    "language":     ["language", "languages"],
}

_GENERIC_QUALIFIERS: frozenset[str] = frozenset({
    "language", "languages", "tool", "tools", "skill", "skills",
    "technology", "technologies", "software", "framework", "concept",
    "concepts", "term", "terms", "field", "area",
})

def _not_covered_msg(file_label: str) -> str:
    return (
        f"That information does not appear in the {file_label}. "
        "Please check the document or rephrase your question."
    )
