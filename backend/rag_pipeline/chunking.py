from __future__ import annotations
import re
import sys

from .common import RELEVANCE_THRESHOLD, _HEADING_RE, _SUBENTRY_RE, _BULLET_RE
from .document_structure import split_numbered_sections

def get_adaptive_threshold(chunk: str, base_threshold: float = RELEVANCE_THRESHOLD) -> float:
    """
    Adaptive cosine distance threshold based on chunk character length.
    Short, telegraphic chunks (e.g. resume bullets, section headings + 1 item)
    have fewer overlapping tokens with conversational questions, so their
    ChromaDB cosine distance to the question embedding is naturally higher
    (often 0.72–0.82). Longer chunks carry more context and are held to the
    stricter base threshold (<= 0.70).
    """
    length = len(chunk.strip())
    if length < 100:
        return max(base_threshold, 0.84)
    elif length < 250:
        return max(base_threshold, 0.80)
    elif length < 500:
        return max(base_threshold, 0.75)
    else:
        return base_threshold

def _normalise_resume_inline_headings(text: str) -> str:
    """Insert deterministic line breaks before strong resume headings embedded inline.

    Some PDFs flatten a visually structured two-column CV into long text lines.
    The existing structure-aware parser works line-by-line, so inline headings such as
    ``PROFESSIONAL EXPERIENCE`` or ``CONTACT`` could disappear into the preceding body.
    We only activate this normalization when the document shows multiple strong resume
    signals, reducing false positives on ordinary prose documents.
    """
    raw = text or ""
    lower = raw.lower()
    strong_heads = (
        "professional experience", "work experience", "employment history",
        "education", "professional summary", "career objective",
        "expertise", "soft skills", "technical skills", "contact",
    )
    signal_count = sum(1 for h in strong_heads if h in lower)
    if signal_count < 2:
        return raw

    result = raw
    # Multi-word/strong headings first. Use boundaries so we do not split inside
    # larger words; require either all-caps form or a preceding newline/large gap.
    heading_patterns = (
        "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT HISTORY",
        "PROFESSIONAL SUMMARY", "CAREER OBJECTIVE", "TECHNICAL SKILLS",
        "SOFT SKILLS", "PERSONAL INFORMATION", "ACADEMIC BACKGROUND",
        "EDUCATION", "EXPERTISE", "CONTACT", "LANGUAGE", "LANGUAGES",
        "REFERENCES", "PROFILE", "ABOUT",
    )
    for heading in heading_patterns:
        pattern = re.compile(rf"(?<![\n])(?<![A-Za-z])({re.escape(heading)})(?=\s|:|$)", re.IGNORECASE)
        def repl(match: re.Match) -> str:
            token = match.group(1)
            # Strong signal: all-caps, title-style line after a large whitespace run,
            # or a known multi-word heading.
            before = raw[max(0, match.start()-4):match.start()]
            # Only split when the heading is visibly strong (all-caps) or is
            # clearly separated like a layout heading. Do not split ordinary
            # prose occurrences such as "education helps...".
            if token.isupper() or "  " in before or "\t" in before:
                return "\n" + token
            return token
        result = pattern.sub(repl, result)

    return result


def _split_large_entry(text: str, max_size: int = 800, overlap: int = 100) -> list[str]:
    """Split a large section entry into manageable chunks, preferring line boundaries."""
    if len(text) <= max_size:
        return [text]
    lines = text.splitlines()
    sub_chunks: list[str] = []
    current: list[str] = []
    cur_len = 0
    for line in lines:
        l_len = len(line) + 1
        if cur_len + l_len > max_size and current:
            sub_chunks.append("\n".join(current).strip())
            if len(line) < overlap and current:
                current = [current[-1], line]
                cur_len = len(current[0]) + 1 + l_len
            else:
                current = [line]
                cur_len = l_len
        else:
            current.append(line)
            cur_len += l_len
    if current:
        sub_chunks.append("\n".join(current).strip())
    return [c for c in sub_chunks if c]

def _numbered_report_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks, _meta = split_numbered_sections(text, chunk_size=chunk_size, overlap=overlap)
    if chunks:
        print(f"[rag] Numbered-section chunking produced {len(chunks)} section-aware chunks.", file=sys.stderr)
    return chunks

def _structure_aware_chunks(text: str) -> list[str]:
    """
    Split *text* into fine-grained, context-preserving chunks for structured
    documents such as resumes, CVs, forms, and reports.

    Strategy
    --------
    1. Detect section headings (EDUCATION, EXPERIENCE, SKILLS …).
    2. Within each section, detect bullet/list entries.
    3. Each bullet entry becomes its own chunk, prefixed with the section
       heading so the chunk is self-contained and semantically specific.

    Example — input:
        EDUCATION
        • Matric
          Computer Science from PAK Angels Grammar School
        • Intermediate
          ICS-PHY From Govt. Islamia College Civil Lines
        • Graduation
          Continued in BBIT

    Example — output chunks:
        "Section: EDUCATION\nMatric\nComputer Science from PAK Angels Grammar School"
        "Section: EDUCATION\nIntermediate\nICS-PHY From Govt. Islamia College Civil Lines"
        "Section: EDUCATION\nGraduation\nContinued in BBIT"

    If no structured headings are detected (e.g. plain prose), returns an
    empty list and the caller falls back to the generic chunker.
    """
    text = _normalise_resume_inline_headings(text)
    lines = text.splitlines()
    chunks: list[str] = []

    current_section = ""
    current_entry_lines: list[str] = []

    def flush_entry() -> None:
        if current_entry_lines:
            entry_text = "\n".join(l.strip() for l in current_entry_lines if l.strip())
            if entry_text:
                prefix = f"Section: {current_section}\n" if current_section else ""
                if len(entry_text) <= 1200:
                    chunks.append(prefix + entry_text)
                else:
                    # Guard: avoid giant unbulleted chunks (e.g. thousands of chars)
                    sub_chunks = _split_large_entry(entry_text, max_size=800, overlap=100)
                    for sc in sub_chunks:
                        chunks.append(prefix + sc)
        current_entry_lines.clear()

    found_heading = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip file header tag if present at start of document
        if stripped.startswith("[FILE:") and stripped.endswith("]"):
            continue

        # Detect section heading (standalone or inline)
        m = _HEADING_RE.match(stripped)
        if m:
            flush_entry()
            current_section = m.group(1).upper()
            found_heading = True
            inline_content = (m.group(2) or "").strip()
            if inline_content:
                if _BULLET_RE.match(inline_content):
                    inline_content = _BULLET_RE.sub("", inline_content).strip()
                current_entry_lines.append(inline_content)
            continue

        # If no heading has been encountered yet, capture into initial PROFILE section
        if not current_section:
            current_section = "PROFILE"
            current_entry_lines.append(stripped)
            continue

        # Detect bullet entry start
        if _BULLET_RE.match(line) or _BULLET_RE.match(stripped):
            flush_entry()
            content = _BULLET_RE.sub("", stripped).strip()
            if content:
                current_entry_lines.append(content)
            continue

        # Indented continuation line for current entry
        if line.startswith(("  ", "\t")) and current_entry_lines:
            current_entry_lines.append(stripped)
            continue

        # Detect known sub-entry within a section (e.g. Matric, Intermediate, Graduation)
        if current_section and _SUBENTRY_RE.match(stripped):
            flush_entry()
            current_entry_lines.append(stripped)
            continue

        # Continuation line for current entry
        if current_entry_lines:
            current_entry_lines.append(stripped)
            continue

        # Non-empty line within a section when no entry has started
        if current_section:
            current_entry_lines.append(stripped)
            continue

    flush_entry()

    # Return structure-aware chunks if we detected at least one heading
    # and produced at least one meaningful chunk.
    # Relaxed from >= 2 to >= 1 so single-section documents (e.g. a
    # contact-only snippet) are still handled by this path.
    if found_heading and len(chunks) >= 1:
        return chunks
    return []

def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size*
    characters, preferring paragraph and sentence boundaries.

    For structured documents (resumes, CVs, forms), a structure-aware
    chunker runs first and produces entry-level chunks.  If it succeeds,
    those chunks are used.  If the document is unstructured prose, the
    generic paragraph/sentence chunker runs instead.

    Parameters
    ----------
    text       : The full document text.
    chunk_size : Target character count per chunk (default 800).
    overlap    : Characters to carry forward from the previous chunk (default 100).

    Returns
    -------
    list[str]  : Non-empty chunk strings.
    """
    if not text or not text.strip():
        return []

    # ── Prefer numbered-report structure when present ───────────
    report_chunks = _numbered_report_chunks(text, chunk_size=chunk_size, overlap=overlap)
    if report_chunks:
        return [c for c in report_chunks if c.strip()]

    # ── Try structure-aware chunking first ─────────────────────
    structured = _structure_aware_chunks(text)
    if structured:
        print(
            f"[rag] Structure-aware chunking produced {len(structured)} entry-level chunks.",
            file=sys.stderr,
        )
        return [c for c in structured if c.strip()]

    # ── Generic paragraph/sentence chunker (unchanged) ─────────
    paragraphs = re.split(r"\n{2,}", text)

    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk = (
                        current_chunk + " " + sentence if current_chunk else sentence
                    )
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    if len(sentence) > chunk_size:
                        for i in range(0, len(sentence), chunk_size - overlap):
                            part = sentence[i : i + chunk_size]
                            if part.strip():
                                chunks.append(part.strip())
                        current_chunk = sentence[-(overlap):] if len(sentence) > overlap else sentence
                    else:
                        current_chunk = sentence
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = (
                    current_chunk + "\n\n" + para if current_chunk else para
                )
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if overlap > 0 and current_chunk:
                    tail = current_chunk[-overlap:]
                    current_chunk = tail + "\n\n" + para
                else:
                    current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]
