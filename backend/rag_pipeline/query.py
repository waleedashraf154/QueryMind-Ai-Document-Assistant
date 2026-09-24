from __future__ import annotations
import os
import re
import sys
from typing import Optional

from .common import _STOP_WORDS, _GENERIC_QUALIFIERS, _BACKEND_DIR
from .document_structure import extract_structural_headings_from_chunks

_ORDINAL_NUMBERS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}

_STRUCTURED_NUMBER_RE = r"(\d+(?:\.\d+)*)"

# Generic document-structure references.  These are labels, not individual
# question hardcodes.  They intentionally support hierarchical numbers such as
# 3.2.1 and natural variants like "point no 3.2.1" / "section 3.2.1".
_STRUCTURED_REFERENCE_PATTERNS = (
    ("question", re.compile(rf"(?<!\w)(?:q|question)\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("chapter", re.compile(rf"(?<!\w)(?:chapter|chap\.)\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("section", re.compile(rf"(?<!\w)(?:section|sec\.?|subsection|sub-section)\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("step", re.compile(rf"(?<!\w)step\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("point", re.compile(rf"(?<!\w)point\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("heading", re.compile(rf"(?<!\w)heading\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("problem", re.compile(rf"(?<!\w)(?:problem|prob\.)\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("exercise", re.compile(rf"(?<!\w)exercise\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("example", re.compile(rf"(?<!\w)example\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("task", re.compile(rf"(?<!\w)task\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("item", re.compile(rf"(?<!\w)item\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])", re.IGNORECASE)),
    ("page", re.compile(r"(?<!\w)(?:page)\s*(?:number|no\.?|#)?\s*(\d+)(?!\d)", re.IGNORECASE)),
    ("part", re.compile(r"(?<!\w)part\s*([A-Za-z])(?![A-Za-z])", re.IGNORECASE)),
)

# Natural phrasing such as "heading of 3.2.2" / "section for 4.1".
_STRUCTURED_OF_PATTERN = re.compile(
    rf"(?<!\w)(question|chapter|section|sec\.?|subsection|sub-section|step|point|heading(?:/title)?|title|name|problem|prob\.?|exercise|example|task|item)"
    rf"\s+(?:of|for)\s*(?:number|no\.?|#)?\s*{_STRUCTURED_NUMBER_RE}(?![\d.])",
    re.IGNORECASE,
)

def _extract_structured_references(question: str) -> list[tuple[str, str]]:
    """Extract generic document-structure references from natural language."""
    q = re.sub(r"\s+", " ", (question or "").strip())
    if not q:
        return []

    found: list[tuple[int, str, str]] = []
    for kind, pattern in _STRUCTURED_REFERENCE_PATTERNS:
        for match in pattern.finditer(q):
            value = match.group(1).upper() if kind == "part" else match.group(1)
            found.append((match.start(), kind, value))

    ordinal_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, _ORDINAL_NUMBERS.keys())) + r")\s+"
        r"(question|chapter|section|step|point|heading|problem|exercise|example|task|item|page)\b",
        re.IGNORECASE,
    )
    for match in ordinal_pattern.finditer(q):
        number = _ORDINAL_NUMBERS.get(match.group(1).lower())
        if number is not None:
            found.append((match.start(), match.group(2).lower(), str(number)))

    for match in _STRUCTURED_OF_PATTERN.finditer(q):
        kind = match.group(1).lower()
        if kind == "sec.":
            kind = "section"
        elif kind == "prob.":
            kind = "problem"
        elif kind in {"heading/title", "title", "name"}:
            kind = "heading"
        found.append((match.start(), kind, match.group(2)))

    found.sort(key=lambda item: item[0])
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, kind, value in found:
        key = (kind, value)
        if key not in seen:
            seen.add(key)
            result.append(key)
        if len(result) >= 4:
            break
    return result

def _has_structured_reference(question: str) -> bool:
    return bool(_extract_structured_references(question))


def _specific_structured_reference(question: str) -> Optional[tuple[str, str]]:
    """Return one concrete structural reference when the query targets one item."""
    refs = _extract_structured_references(question)
    if len(refs) != 1:
        return None
    if _is_heading_extraction_query(question):
        return None
    return refs[0]


def _is_specific_structured_query(question: str) -> bool:
    return _specific_structured_reference(question) is not None


def _is_structured_heading_title_request(question: str) -> bool:
    """Return True when the user wants the title/name of one referenced structure item."""
    ref = _specific_structured_reference(question)
    if not ref:
        return False
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    return bool(
        re.search(r"\b(?:heading|title|name)\b.*\b(?:of|for)\b", q)
        or re.search(r"\bwhat\s+is\s+(?:the\s+)?(?:heading|title|name)\b", q)
    )


def _extract_definition_term(question: str) -> Optional[str]:
    """Extract the term from a genuine definition/meaning question.

    Distinguishes genuine conceptual definition inquiries ("Define Naive Bayes",
    "What does Naive Bayes mean?", "What is the definition of Naive Bayes?",
    "What is Naive Bayes?") from document questions asking for specific
    document content, attributes, values, or answers ("What is the final answer of Q1?",
    "What formula is used?", "What is the value of temperature?", "What is the candidate's
    education?").
    """
    q = re.sub(r"\s+", " ", question.strip())
    q = re.sub(r"[?!]+$", "", q).strip()

    # If the user question references structured sections (e.g. Q1, step 3, section 2),
    # it is a document-structure query, not a generic definition.
    if _has_structured_reference(q):
        return None

    # Patterns for explicit definition requests
    explicit_patterns = [
        r"^(?:what\s+is|what\s+are)\s+(?:the\s+)?(?:meaning|definition)\s+of\s+(.+?)$",
        r"^(?:define|definition\s+of|meaning\s+of)\s+(?:an?\s+|the\s+)?(.+?)$",
        r"^what\s+does\s+(.+?)\s+mean(?:ing)?$",
    ]

    # Pattern for general "what is [concept]?"
    concept_pattern = r"^(?:what\s+is|what\s+are)\s+(?:an?\s+|the\s+)?(.+?)$"

    term = None
    is_explicit = False

    for pattern in explicit_patterns:
        m = re.match(pattern, q, flags=re.IGNORECASE)
        if m:
            term = m.group(1).strip(" \t,.:;-")
            is_explicit = True
            break

    if not term:
        m = re.match(concept_pattern, q, flags=re.IGNORECASE)
        if m:
            term = m.group(1).strip(" \t,.:;-")

    if not term:
        return None

    lower = term.lower().strip()

    # Clean off conversational suffixes
    lower = re.sub(
        r"\s+(?:please|for me|in simple words|in simple terms|briefly)$",
        "",
        lower,
    ).strip(" \t,.:;-")

    # Document-fact indicators: questions asking about values, results, formulas,
    # predictions, or calculations are document QA, not dictionary definitions.
    fact_indicators = (
        r"\b(answer|solution|value|values|formula|formulas|equation|calculation|"
        r"result|results|prediction|predictions|temperature|metric|metrics|score|scores|"
        r"total|sum|average|count|difference|status|date|time|output|input|cost|price|"
        r"percentage|rate|probability|probabilities|reason|reasons|step|steps)\b"
    )
    if re.search(fact_indicators, lower):
        return None

    # Document container/reference words
    if re.search(
        r"\b(document|documents|file|files|pdf|pdfs|docx|pptx|page|pages|slide|slides|"
        r"sheet|sheets|table|tables|chart|charts|diagram|diagrams|image|images|figure|figures|"
        r"text|section|sections|chapter|chapters|line|lines|paragraph|paragraphs)\b",
        lower,
    ):
        return None

    # Passive verbs indicating document locations ("mentioned under", "written in", "stated in")
    if re.search(
        r"\b(shown|depicted|written|visible|seen|displayed|described|stated|found|included|"
        r"mentioned|used|given|provided|listed|calculated|predicted)\b",
        lower,
    ):
        return None

    # Prepositional location phrases ("in the", "from the", "under", "according to")
    if re.search(r"\b(?:in|from|according to|as per|under|at)\s+(?:the|this|a|an|our)\b", lower):
        return None

    # Demonstratives, pronouns, deictics
    if lower in {
        "this", "that", "these", "those", "it", "here", "there",
        "something", "anything", "everything", "nothing",
    }:
        return None

    # Personal or document entities (e.g. "candidate's education", "your experience")
    if re.search(r"\b(your|my|our|their|his|her|candidate(?:’s|'s|s)?|person(?:’s|'s|s)?|applicant(?:’s|'s|s)?)\b", lower):
        return None
    if re.search(
        r"\b(education|qualification|qualifications|degree|experience|job|role|position|"
        r"company|employer|name|email|phone|address|salary|age|dob|date\s+of\s+birth|"
        r"nationality|cnic|language|languages|skill|skills|hobby|hobbies)\b$",
        lower,
    ):
        return None

    # Non-explicit "what is X" questions with more than 4 words are almost always
    # descriptive questions rather than conceptual definitions.
    if not is_explicit and len(lower.split()) > 4:
        return None

    # Non-explicit "what is the X of Y?" questions are attribute/property queries,
    # not definitions.  The universal topic gate in document_qa mode handles these
    # correctly by extracting the actual subject (Y) rather than the attribute phrase.
    # Example: "What is the color of biryani?" → return None here so mode=document_qa
    # and the gate extracts topic="biryani" from the LLM router.
    if not is_explicit and re.search(r"\s+of\s+\w", lower):
        return None

    if len(lower) < 2 or len(lower) > 100:
        return None

    return term

def _extract_topic_term(question: str) -> Optional[str]:
    """Extract a topic from open-ended 'tell me about/explain' questions.

    These questions are allowed to use general knowledge only after the topic
    itself has been verified as present in the uploaded document.
    Document-fact questions such as "What is the nationality of the candidate?"
    are deliberately excluded and continue through normal RAG.
    """
    q = re.sub(r"\s+", " ", question.strip())
    q = re.sub(r"[?!]+$", "", q).strip()

    patterns = [
        r"^(?:tell me about|what can you tell me about|explain|describe)\s+(.+?)$",
    ]
    term = None
    for pattern in patterns:
        m = re.match(pattern, q, flags=re.IGNORECASE)
        if m:
            term = m.group(1).strip(" \\t,.:;-")
            break
    if not term:
        return None

    lower = term.lower()

    if lower in {
        "this", "that", "these", "those", "it", "here", "there",
        "image", "images", "picture", "pictures", "photo", "photos", "screenshot", "screenshots",
        "document", "documents", "file", "files", "pdf", "pdfs", "page", "pages",
        "something", "anything", "everything", "nothing",
    }:
        return None

    # Personal/document-specific requests must remain document-grounded.
    if re.search(r"\b(your|my|our|their|his|her|candidate(?:’s|'s|s)?|person(?:’s|'s|s)?)\b", lower):
        return None
    # Don't treat obvious document fields as general-knowledge topics when
    # the user is asking for the candidate's actual value.
    if re.search(r"\b(nationality|education|qualification|degree|experience|job|role|position|company|employer|name|email|phone|address|salary|age|dob|date of birth)\b", lower):
        return None

    term = re.sub(r"\s+(?:please|for me|in simple words|in simple terms|briefly)$", "", term, flags=re.IGNORECASE).strip(" \\t,.:;-")
    if len(term) < 2 or len(term) > 120:
        return None
    return term


# ---------------------------------------------------------------------------
# Core entity / topic words that are too generic to act as topic anchors.
# These are filtered out when extracting the informational topic from a query.
# ---------------------------------------------------------------------------
_INFO_TOPIC_STOP_WORDS: frozenset[str] = frozenset({
    # question words
    "who", "what", "where", "when", "why", "how", "which", "whose",
    # articles / prepositions
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "into", "about", "through", "during", "before", "after",
    # common verbs / auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "has", "have",
    "had", "let", "make", "made", "get", "got", "give", "given",
    # common generic nouns (usually not the real topic)
    "color", "colour", "name", "number", "amount", "type", "kind", "sort",
    "information", "detail", "details", "thing", "things", "fact", "facts",
    "way", "reason", "reasons", "example", "examples",
    # pronouns
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "their", "him", "her", "this", "that", "these", "those",
    # filler / conversational
    "please", "just", "really", "also", "so", "then", "there", "here",
    "any", "some", "many", "much", "more", "most", "other", "another",
    "all", "both", "each", "every", "no", "not", "only", "own", "same",
    "such", "too", "very", "well", "rather", "quite", "already", "still",
})

# Patterns where the question asks about an ATTRIBUTE of an entity.
# "What is the color of biryani?" → we want "biryani", not "color".
# "What is the capital of France?" → we want "France".
_ATTRIBUTE_OF_PATTERN = re.compile(
    r"^(?:what|where|when|who|why|how)\s+(?:is|are|was|were)\s+"
    r"(?:the|a|an)\s+[\w\s]+?\s+of\s+(.+)$",
    re.IGNORECASE,
)

# Patterns for "who is/was X", "what is X", "how does X work" etc.
_WH_SIMPLE_PATTERN = re.compile(
    r"^(?:who|what)\s+(?:is|are|was|were)\s+(?:a|an|the\s+)?([\w/][\w\s\-'/]{1,80})$",
    re.IGNORECASE,
)

# "who founded / created / invented / built X"
_WH_VERB_OBJECT_PATTERN = re.compile(
    r"^(?:who|what)\s+\w+(?:ed|d|t)?\s+(?:a|an|the\s+)?([\w/][\w\s\-'/]{1,80})$",
    re.IGNORECASE,
)

# "what did/does/has X do/release/produce/..."
# "what albums has Zayn Malik released?"
# NOTE: {0,2} prefix limit avoids consuming multi-word proper nouns like "Zayn Malik".
_WHAT_DID_PATTERN = re.compile(
    r"^what\s+(?:\w+\s+){0,2}([\w/][\w\s\-'/]{1,60})\s+(?:do|does|did|has|have|release[d]?|produce[d]?|make|made|achieve[d]?|win|won|say|said|write|written|publish[ed]?|include[d]?).*$",
    re.IGNORECASE,
)


def _extract_informational_topic(question: str) -> Optional[str]:
    """Extract the core entity/topic from ANY information-seeking question.

    This is the universal catch-all for the document_qa path.  It runs AFTER
    definition and topic modes have already been tried, so it only needs to
    handle the patterns those modes miss:

      "Who is Zayn Malik?"            → "Zayn Malik"
      "What is the color of biryani?" → "biryani"
      "Who founded Pakistan?"         → "Pakistan"
      "What albums has Zayn Malik released?" → "Zayn Malik"
      "What is Python?"               → "Python"  (also caught by definition, but harmless)

    Returns None for:
      - Pure social messages (no information-seeking intent)
      - Document-structure queries (Q1, section 2, page 3)
      - Personal/document-field queries (candidate's X, your X)
      - Questions too vague to pin to a specific entity
    """
    if not question:
        return None

    q = re.sub(r"\s+", " ", question.strip())
    q_clean = re.sub(r"[?!.]+$", "", q).strip()

    # Skip document-structure, summary, and section queries — they are grounded in the doc by structure.
    if _has_structured_reference(q_clean) or _is_summary_query(q_clean) or _is_heading_extraction_query(q_clean) or _parse_section_target(q_clean):
        return None

    q_lower = q_clean.lower()

    # Broad/multi-hop document questions must go directly to RAG. Treating an
    # entire relational phrase as a "topic" can cause the topic gate to reject
    # a question before semantic retrieval gets a chance to find the evidence.
    if re.search(
        r"\b(?:relationship|relate|related|connection|compare|comparison|difference|versus|\bvs\b|"
        r"between|reason|why|purpose|impact|effect|process|workflow|how\s+does|how\s+do|"
        r"according\s+to|based\s+on|from\s+the\s+(?:file|document|pdf)|explain|describe)\b",
        q_lower,
    ):
        return None

    # Skip pure social messages — not information-seeking.
    _social_re = re.compile(
        r"^(?:hi|hello|hey|yo|howdy|greetings|good\s+(?:morning|afternoon|evening|night)|"
        r"thanks?|thank\s+you|cheers|bye|goodbye|see\s+you|ok|okay|got\s+it|sure|"
        r"sounds\s+good|nice|great|awesome|cool|alright|right|fine|no\s+problem|"
        r"how\s+are\s+you|how\s+r\s+u|what['']?s\s+up|wassup|whats\s+up)[\s!.,]*$",
        re.IGNORECASE,
    )
    if _social_re.match(q_clean):
        return None

    # Skip personal / document-field queries.
    _personal_re = re.compile(
        r"\b(your|my|our|their|his|her|candidate(?:'s)?|person(?:'s)?|applicant(?:'s)?)\b",
        re.IGNORECASE,
    )
    if _personal_re.search(q_clean):
        return None
    _field_re = re.compile(
        r"\b(nationality|education|qualification|degree|experience|job|role|position|"
        r"company|employer|email|phone|address|salary|age|dob|date\s+of\s+birth)\b",
        re.IGNORECASE,
    )
    if _field_re.search(q_lower):
        return None

    # ── Strategy 1: "What is the <attr> of <entity>?" ──────────────────────
    m = _ATTRIBUTE_OF_PATTERN.match(q_clean)
    if m:
        candidate = m.group(1).strip(" ,.:;-'\"")
        candidate = re.sub(r"\s+", " ", candidate)
        if candidate and len(candidate) >= 2:
            print(
                f"[topic] extracted_topic='{candidate}' (via attribute-of pattern)",
                file=sys.stderr,
            )
            return candidate

    # ── Strategy 2: "Who/What is/are/was/were [a/an/the] <entity>?" ────────
    m = _WH_SIMPLE_PATTERN.match(q_clean)
    if m:
        candidate = m.group(1).strip(" ,.:;-'\"")
        candidate = re.sub(r"\s+", " ", candidate)
        words = candidate.split()
        # Filter trailing stop words
        while words and words[-1].lower() in _INFO_TOPIC_STOP_WORDS:
            words.pop()
        candidate = " ".join(words)
        if candidate and len(candidate) >= 2 and candidate.lower() not in _INFO_TOPIC_STOP_WORDS:
            print(
                f"[topic] extracted_topic='{candidate}' (via wh-simple pattern)",
                file=sys.stderr,
            )
            return candidate

    # ── Strategy 3: "What did/does/has <entity> <verb>?" ───────────────────
    m = _WHAT_DID_PATTERN.match(q_clean)
    if m:
        candidate = m.group(1).strip(" ,.:;-'\"")
        candidate = re.sub(r"\s+", " ", candidate)
        # Remove leading stop words
        words = candidate.split()
        while words and words[0].lower() in _INFO_TOPIC_STOP_WORDS:
            words.pop(0)
        while words and words[-1].lower() in _INFO_TOPIC_STOP_WORDS:
            words.pop()
        candidate = " ".join(words)
        if candidate and len(candidate) >= 2 and candidate.lower() not in _INFO_TOPIC_STOP_WORDS:
            print(
                f"[topic] extracted_topic='{candidate}' (via what-did pattern)",
                file=sys.stderr,
            )
            return candidate

    # ── Strategy 4a: questions ending with a clear prepositional entity ──────
    # Examples: "What ingredients are used in biryani?" -> "biryani".
    m_prep = re.search(r"\b(?:in|for|at|from|about|under)\s+([A-Za-z0-9][\w/\s\-'/]{1,80})$", q_clean, re.IGNORECASE)
    if m_prep:
        candidate = m_prep.group(1).strip(" ,.:;-'\"")
        words = candidate.split()
        while words and words[-1].lower() in _INFO_TOPIC_STOP_WORDS:
            words.pop()
        candidate = " ".join(words)
        if candidate and len(candidate) >= 2 and candidate.lower() not in _INFO_TOPIC_STOP_WORDS:
            print(
                f"[topic] extracted_topic='{candidate}' (via prepositional-entity pattern)",
                file=sys.stderr,
            )
            return candidate

    # ── Strategy 4: "Who <verb> <entity>?" ─────────────────────────────────
    m = _WH_VERB_OBJECT_PATTERN.match(q_clean)
    if m:
        candidate = m.group(1).strip(" ,.:;-'\"")
        candidate = re.sub(r"\s+", " ", candidate)
        words = candidate.split()
        while words and words[-1].lower() in _INFO_TOPIC_STOP_WORDS:
            words.pop()
        candidate = " ".join(words)
        if candidate and len(candidate) >= 2 and candidate.lower() not in _INFO_TOPIC_STOP_WORDS:
            print(
                f"[topic] extracted_topic='{candidate}' (via wh-verb-object pattern)",
                file=sys.stderr,
            )
            return candidate

    # ── Strategy 5: Extract the longest run of capitalized words (named entity) ──
    # "What albums has Zayn Malik released?" → "Zayn Malik"
    # Only applies when question has >= 2 words (avoid matching sentence-start capital)
    tokens = q_clean.split()
    if len(tokens) >= 3:
        # Collect all capitalized multi-word runs (skip first token — often a question word)
        runs: list[str] = []
        current_run: list[str] = []
        for tok in tokens[1:]:
            bare = tok.strip(",.?!;:'\"()")
            if bare and bare[0].isupper() and bare.lower() not in _INFO_TOPIC_STOP_WORDS:
                current_run.append(bare)
            else:
                if len(current_run) >= 1:
                    runs.append(" ".join(current_run))
                current_run = []
        if current_run:
            runs.append(" ".join(current_run))

        # Pick the longest run that is not a generic word
        if runs:
            best = max(runs, key=len)
            if len(best) >= 2 and best.lower() not in _INFO_TOPIC_STOP_WORDS:
                print(
                    f"[topic] extracted_topic='{best}' (via capitalized named-entity heuristic)",
                    file=sys.stderr,
                )
                return best

    return None

def _normalise_for_term_search(text: str) -> str:
    """Normalize punctuation, hyphens and possessives for reliable term matching."""
    text = text.lower().replace("&", " and ")
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _term_variants(term: str) -> list[str]:
    """Return safe variants for common singular/plural wording differences."""
    target = _normalise_for_term_search(term)
    if not target:
        return []

    variants = [target]
    compact = target.replace(" ", "")
    if compact != target and len(compact) > 3:
        variants.append(compact)

    words = target.split()
    last = words[-1]

    # Handle common English pluralization so e.g. "observation" can match
    # "observations" in a document, without doing broad semantic matching.
    if len(last) > 3:
        if last.endswith("y") and not last.endswith(("ay", "ey", "iy", "oy", "uy")):
            variants.append(" ".join(words[:-1] + [last[:-1] + "ies"]))
        elif last.endswith(("s", "x", "z", "ch", "sh")):
            variants.append(" ".join(words[:-1] + [last + "es"]))
        else:
            variants.append(" ".join(words[:-1] + [last + "s"]))

        # Also try singular when the user asks with a plural form.
        if last.endswith("ies") and len(last) > 4:
            variants.append(" ".join(words[:-1] + [last[:-3] + "y"]))
        elif last.endswith("es") and len(last) > 4:
            variants.append(" ".join(words[:-1] + [last[:-2]]))
        elif last.endswith("s") and len(last) > 4:
            variants.append(" ".join(words[:-1] + [last[:-1]]))

    return list(dict.fromkeys(v for v in variants if v))

def _find_topic_in_entire_document(
    collection,
    topic: str,
    target_source: Optional[str] = None,
) -> tuple[bool, list[str], list[dict], str]:
    """Find an open-ended topic anywhere in the indexed document.

    For phrases such as "Urdu language", the document may contain only
    "Urdu" (e.g. under a LANGUAGES section). In that case the distinctive
    topic token is enough to verify the topic. Generic words such as
    "language", "about", "tell", etc. are ignored.
    """
    normalized_topic = _normalise_for_term_search(topic)
    if not normalized_topic:
        return False, [], [], ""

    stop = _STOP_WORDS | {
        "about", "tell", "please", "explain", "describe", "language",
        "definition", "meaning", "information", "details", "thing",
    }
    # Use >= 2 (not > 2) so 2-letter acronyms like "ai", "ml" are included.
    tokens = [t for t in normalized_topic.split() if len(t) >= 2 and t not in stop]
    if not tokens:
        return False, [], [], ""

    phrase_pattern = re.compile(rf"(?<!\w){re.escape(normalized_topic)}(?!\w)", re.IGNORECASE)
    token_patterns = [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE) for t in tokens]

    matched_docs, matched_metas = [], []
    matched_by = ""
    try:
        data = collection.get(include=["documents", "metadatas"])
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        for doc, meta in zip(docs, metas or [{}] * len(docs)):
            if target_source and (meta or {}).get("source") != target_source:
                continue
            norm_doc = _normalise_for_term_search(doc or "")
            if phrase_pattern.search(norm_doc):
                matched_docs.append(doc)
                matched_metas.append(meta or {})
                matched_by = "phrase"
            elif (
                (len(token_patterns) == 1 and token_patterns[0].search(norm_doc))
                or (len(token_patterns) > 1 and all(p.search(norm_doc) for p in token_patterns))
            ):
                matched_docs.append(doc)
                matched_metas.append(meta or {})
                if not matched_by:
                    matched_by = "tokens"
    except Exception as exc:
        print(f"[rag] topic scan failed: {exc}", file=sys.stderr)

    return bool(matched_docs), matched_docs, matched_metas, matched_by

def _find_term_in_entire_document(
    chat_id: str,
    collection,
    term: str,
    target_source: Optional[str] = None,
) -> tuple[bool, list[str], list[dict], str]:
    """Verify a definition term against the COMPLETE uploaded document.

    This deliberately does not depend only on vector retrieval or the top-K
    chunks. It checks both:
      1. every indexed Chroma chunk (filtered by target_source if specified), and
      2. the original uploaded files stored for this chat.

    Handles natural variations:
      - case-insensitive normalized matching
      - singular / plural variations
      - trailing / leading category qualifiers (e.g. 'Urdu Language' -> matches 'Urdu')
      - hyphen and whitespace variations (e.g. 'Multi-Tasking' <-> 'multitasking')
    """
    candidate_terms = [term]
    words = [w for w in term.strip().split() if w]
    if len(words) > 1:
        reduced = [w for w in words if w.lower() not in _GENERIC_QUALIFIERS]
        if reduced and len(reduced) < len(words):
            candidate_terms.append(" ".join(reduced))

    for candidate in candidate_terms:
        variants = _term_variants(candidate)
        if not variants:
            continue

        patterns = [
            re.compile(rf"(?<!\w){re.escape(v)}(?!\w)", re.IGNORECASE)
            for v in variants
        ]

        def matches(text: str) -> bool:
            norm = _normalise_for_term_search(text or "")
            compact = norm.replace(" ", "")
            return any(pattern.search(norm) or pattern.search(compact) for pattern in patterns)

        matched_docs: list[str] = []
        matched_metas: list[dict] = []
        seen_sources: set[str] = set()

        # -------------------------------------------------------------
        # 1. Search ALL indexed chunks, not just retrieved chunks.
        # -------------------------------------------------------------
        try:
            data = collection.get(include=["documents", "metadatas"])
            docs = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []

            for doc, meta in zip(docs, metas or [{}] * len(docs)):
                if target_source and (meta or {}).get("source") != target_source:
                    continue
                if matches(doc or ""):
                    matched_docs.append(doc)
                    matched_metas.append(meta or {})
                    src = (meta or {}).get("source", "")
                    if src:
                        seen_sources.add(src)
        except Exception as exc:
            print(f"[rag] indexed-document term scan failed: {exc}", file=sys.stderr)

        if matched_docs:
            print(
                f"[rag] DEFINITION TERM FOUND: '{candidate}' (from '{term}') in indexed document.",
                file=sys.stderr,
            )
            return True, matched_docs, matched_metas, candidate
            return True, matched_docs, matched_metas, candidate

    print(f"[rag] DEFINITION TERM NOT FOUND: '{term}'", file=sys.stderr)
    return False, [], [], ""

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

def is_image_file(filename: str) -> bool:
    """Return True if filename has an image extension."""
    return os.path.splitext(filename or "")[1].lower() in IMAGE_EXTENSIONS

def _is_visual_intent(question: str) -> bool:
    """Check if the question expresses visual perception or image-related intent.

    Matches visual nouns (image, picture, photo, screenshot, diagram, chart, graphic, etc.)
    and visual observation verbs/deictics (visible, shown, look at, see here, display, etc.)
    without relying on fixed phrase templates.
    """
    q = re.sub(r"\s+", " ", question.strip().lower())

    visual_nouns = r"\b(?:image|images|picture|pictures|photo|photos|photograph|photographs|screenshot|screenshots|diagram|diagrams|chart|charts|graphic|graphics|illustration|illustrations|figure|figures|pic|pics)\b"
    if re.search(visual_nouns, q):
        return True

    visual_perception = r"\b(?:visible|shown|shows|show|displayed|depicted|depicts|see|look\s+at|looking\s+at|view|visual|visually|written\s+here|read\s+this)\b"
    if re.search(visual_perception, q):
        return True

    return False

def _is_image_content_query(question: str) -> bool:
    """Return True when the user is asking to read/extract/describe image content."""
    return _is_visual_intent(question)

def _is_plural_image_content_query(question: str) -> bool:
    """Return True when the user explicitly refers to multiple images/pictures."""
    q = re.sub(r"\s+", " ", question.strip().lower())
    return bool(re.search(r"\b(?:images|pictures|photos|photographs|screenshots|pics)\b", q))




_REFERENCE_TERMS = (
    "reference", "references", "bibliography", "bibliographies",
    "works cited", "sources", "reference list",
)

def _is_reference_count_query(question: str) -> bool:
    """Detect a document-wide request to count bibliography/reference entries.

    This is intentionally semantic/pattern based rather than tied to one exact
    sentence. It is used for deterministic document-wide counting so a top-k RAG
    result cannot produce partial counts.
    """
    q = re.sub(r"\s+", " ", (question or "").strip().casefold())
    if not q:
        return False
    has_ref = any(term in q for term in _REFERENCE_TERMS)
    if not has_ref:
        return False
    count_patterns = (
        r"\bhow\s+many\b",
        r"\b(?:count|number|total|quantity)\b.*\b(?:reference|bibliograph|source)",
        r"\b(?:reference|bibliograph|source)\b.*\b(?:count|number|total|many)\b",
    )
    return any(re.search(p, q, re.IGNORECASE) for p in count_patterns)


def _is_reference_list_query(question: str) -> bool:
    """Detect a request to list/show the document's bibliography/reference entries."""
    q = re.sub(r"\s+", " ", (question or "").strip().casefold())
    if not q or _is_reference_count_query(q):
        return False
    has_ref = any(term in q for term in _REFERENCE_TERMS)
    if not has_ref:
        return False
    return bool(re.search(
        r"\b(?:list|show|display|give|provide|extract|read|all|every|complete|full)\b.*"
        r"\b(?:references?|bibliograph(?:y|ies)|works\s+cited|sources|reference\s+list)\b",
        q, re.IGNORECASE,
    ))

def _is_informational_query(question: str) -> bool:
    """Detect whether a message seeks information rather than pure social chat."""
    q = (question or "").strip().lower()
    _social_re = re.compile(
        r"^(?:hi|hello|hey|howdy|greetings?|good\s+(?:morning|afternoon|evening|night|day)|"
        r"thanks?|thank\s+you|cheers|bye|goodbye|see\s+you|ok|okay|got\s+it|sure|"
        r"sounds\s+good|nice|great|awesome|cool|alright|right|fine|no\s+problem|"
        r"how\s+are\s+you(?:\s+doing)?|how\s+r\s+u|what['']?s\s+up|wassup|whats\s+up|"
        r"how['']?s\s+(?:everything|it\s+going|things|life))[\s!.,?]*$",
        re.IGNORECASE,
    )
    if _social_re.match(q):
        return False

    info_patterns = [
        r"^(?:who|what|where|when|why|which|whose)\b",
        r"^how\s+(?:much|many|does|do|did|can|could|is|are|was|were|to)\b",
        r"\b(?:explain|describe|define|definition|meaning|tell me|list|give me|show me|summarize|summary|overview)\b",
        r"\b(?:who is|what is|what are|where is|when was|who was)\b",
    ]
    return any(re.search(p, q, re.IGNORECASE) for p in info_patterns)


def _has_specific_numbered_heading_reference(question: str) -> bool:
    """Return True when the message refers to one concrete numbered heading.

    This intentionally ignores the requested action. The action is handled by
    downstream routing: "tell me heading 13" can be a title lookup, while
    "explain heading 13" is a section-content request. Both must avoid the
    global "all headings" mode.
    """
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    q = re.sub(r"[?!.]+$", "", q).strip()
    if not q:
        return False
    if re.search(r"\b(?:all|every)\b", q):
        return False
    return bool(
        re.search(r"\bheading\s*(?:number|no\.?|#)?\s*\d+(?:\.\d+)*(?:st|nd|rd|th)?\b", q)
        or re.search(r"\b\d+(?:\.\d+)*(?:st|nd|rd|th)?\s+heading\b", q)
    )


def _is_specific_heading_lookup_query(question: str) -> bool:
    """Detect a request for ONE particular numbered heading, not the whole outline.

    Examples:
      - "Tell me heading 13"
      - "Show me heading no. 13"
      - "What is heading number 13?"

    Explanatory requests such as "explain heading 13" are intentionally left to
    section extraction because the user is asking about the section's content.
    """
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    q = re.sub(r"[?!.]+$", "", q).strip()
    if not q:
        return False

    if re.search(r"\b(?:all|every|main|major|primary|numbered|numbering|outline|table of contents)\b", q):
        return False
    if re.search(r"\b(?:explain|describe|summari[sz]e|meaning|definition|details?|content|about|in)\b", q):
        return False

    specific_patterns = (
        r"\bheading\s*(?:number|no\.?|#)?\s*\d+(?:\.\d+)*(?:st|nd|rd|th)?\b",
        r"\b\d+(?:\.\d+)*(?:st|nd|rd|th)?\s+heading\b",
    )
    if not any(re.search(p, q) for p in specific_patterns):
        return False

    return bool(re.search(r"\b(?:tell|show|give|list|provide|display|extract|get|read|what|which|find)\b", q))


def _is_heading_extraction_query(question: str) -> bool:
    """Detect queries asking for ALL headings, sections, or a table of contents.

    A singular numbered request such as "Tell me heading 13" is deliberately
    excluded so it can be handled as a single-heading lookup.
    """
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    q = re.sub(r"[?!.]+$", "", q).strip()

    # Never treat a concrete structural reference as a request for the entire
    # outline. This includes natural connectors such as "heading of 3.2.2".
    # We use the low-level parser directly here to avoid circular routing calls.
    refs = _extract_structured_references(q)
    if refs and len(refs) == 1 and refs[0][0] != "page":
        return False
    if _has_specific_numbered_heading_reference(q):
        return False

    # A request mentioning headings plus a list/display/count action is a
    # heading-extraction request even when the phrasing is novel (for example
    # "Tell me 15 headings" or "headings only, please").
    if re.search(r"\bheadings?\b", q):
        if re.search(r"\b(?:all|every|main|major|primary|numbered|numbering|only|just|list|show|display|give|tell|provide|extract|get|read|what\s+are|\d+)\b", q):
            return True

    # Specific section/step/chapter/point/heading number queries are NOT
    # heading extraction.
    if re.search(r"\b(?:section|subsection|sub-section|chapter|step|part|item|point|heading|no\.?|number|sec\.?)\s*(?:number|no\.?|#)?\s*\d+(?:\.\d+)*\b", q):
        return False

    patterns = [
        # Normal natural-language forms.
        r"\b(?:give\s+me|list|show|tell\s+me|what\s+are|display|extract|get|provide|read)\s+(?:all\s+|every\s+|the\s+)*(?:main\s+|major\s+|primary\s+|numbered\s+|numbering\s+)*(?:headings?|subheadings?|sections?|section\s+titles?|chapter\s+titles?|chapters?|table\s+of\s+contents?|outline)\b",
        r"\b(?:list\s+every\s+section|all\s+(?:main\s+|major\s+|primary\s+|numbered\s+|numbering\s+)?headings|all\s+sections?|all\s+chapters?|document\s+headings?|document\s+sections?|document\s+outline)\b",
        r"\bwhat\s+are\s+(?:all\s+)?(?:the\s+)?(?:document\s+)?(?:headings?|sections?|chapters?)\b",
        r"\b(?:main|all|numbered|numbering)\s+(?:headings?|sections?|chapters?)\b",
        # Explicit variants used in conversation/UI testing, including
        # “Only Numbering Headings” and “Tell me all 15 headings”.
        r"\b(?:only|just)\s+(?:the\s+)?(?:numbered|numbering|main|major|primary)?\s*headings?\b",
        r"\b(?:tell\s+me|give\s+me|show\s+me|list|provide|display|extract|get)\s+(?:all|every|the)\s+\d+\s+(?:numbered\s+|numbering\s+)?headings?\b",
        r"\ball\s+\d+\s+(?:numbered\s+|numbering\s+)?headings?\b",
        r"\b(?:numbered|numbering)\s+headings?\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _clean_section_target_text(target: str) -> str:
    """Strip conversational document wrappers from a section name."""
    target = re.sub(r"\s+", " ", (target or "").strip())
    # These suffixes describe the source, not the section title.
    target = re.sub(
        r"\s+(?:from|in|inside|within)\s+(?:the\s+)?(?:uploaded\s+)?(?:file|document|pdf|paper|report|text)(?:\s+only)?$",
        "", target, flags=re.IGNORECASE,
    ).strip()
    target = re.sub(r"\s+(?:only|please)$", "", target, flags=re.IGNORECASE).strip()
    target = re.sub(r"^(?:the\s+)?(?:uploaded\s+)?(?:file|document|pdf|paper|report|text)\s*[:\-–—]\s*", "", target, flags=re.IGNORECASE).strip()
    target = re.sub(r"^(?:section|chapter|step|part)\s*[:\-–—]\s*", "", target, flags=re.IGNORECASE).strip()
    return target.strip(" \t,.:;-–—")


_CONTINUATION_RE = re.compile(
    r"^(?:(?:tell|show|give)\s+me\s+)?(?:(?:the|and)\s+)?(?:"
    r"remaining(?:\s+(?:more|part|details))?|"
    r"rest(?:\s+of\s+(?:it|the\s+answer|this))?|"
    r"tell\s+me\s+more|more(?:\s+details)?|continue(?:\s+the\s+(?:answer|section))?|"
    r"go\s+on|finish|complete(?:\s+(?:it|this))?|what\s+else|anything\s+else"
    r")[\s!,.?]*$",
    re.IGNORECASE,
)


def _is_continuation_query(question: str) -> bool:
    q = re.sub(r"\s+", " ", (question or "").strip())
    return bool(_CONTINUATION_RE.match(q))


def _resolve_continuation_question(question: str, conversation_history: Optional[list[dict]] = None) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve short follow-up messages into a retrieval-ready question.

    Returns (working_question, previous_user_question, previous_assistant_answer).
    Retrieval is always performed again against the document; history only supplies
    conversational meaning.
    """
    if not _is_continuation_query(question) or not conversation_history:
        return question, None, None

    current_norm = re.sub(r"\s+", " ", (question or "").strip()).casefold()
    previous_user_idx: Optional[int] = None
    for idx in range(len(conversation_history) - 1, -1, -1):
        item = conversation_history[idx] or {}
        if str(item.get("role", "")).lower() != "user":
            continue
        content = re.sub(r"\s+", " ", str(item.get("content", "") or "").strip()).casefold()
        if content and content != current_norm:
            previous_user_idx = idx
            break

    if previous_user_idx is None:
        return question, None, None

    previous_user = str(conversation_history[previous_user_idx].get("content", "") or "").strip()
    previous_assistant: Optional[str] = None
    for idx in range(previous_user_idx + 1, len(conversation_history)):
        item = conversation_history[idx] or {}
        if str(item.get("role", "")).lower() == "assistant":
            content = str(item.get("content", "") or "").strip()
            if content:
                previous_assistant = content
                break

    parsed = _parse_section_target(previous_user)
    if parsed:
        target = _extract_target_section(previous_user) or "the same section"
        working = (
            f"Continue the previous request about the document section '{target}'. "
            "Retrieve the section again from the uploaded document and provide additional details "
            "not already covered in the previous answer. Do not claim the information is absent "
            "without checking the complete section."
        )
    else:
        working = (
            f"Continue the previous document question: {previous_user}. "
            "Retrieve fresh evidence from the uploaded document and provide additional relevant "
            "details that were not covered in the previous answer."
        )
    return working, previous_user, previous_assistant



def _parse_section_target(question: str) -> Optional[dict]:
    """Extract and normalize target section references (both numbers and titles).

    Supports:
      Numbers:
        - "Tell me step 5", "Tell me 5?", "Tell me 5", "Tell me section 5", "Show me section 5"
        - "What is in section 5?", "Tell me 15", "Give me section 15", "step 5", "section 5"
        -> {"type": "number", "number": "5", "title": None, "raw": "step 5"}
      Titles:
        - "Tell me about Security and Access Rules", "Give me Conclusion and Test Notes"
        - "Tell me Conclusion and Test Notes", "What is in the Executive Overview?"
        -> {"type": "title", "number": None, "title": "Security and Access Rules", "raw": "Security and Access Rules"}
      Number + Title:
        - "Tell me section 5: Security and Access Rules"
        -> {"type": "number_and_title", "number": "5", "title": "Security and Access Rules", "raw": "..."}
    """
    clean = re.sub(r"\s+", " ", (question or "").strip()).rstrip("?!.")
    if not clean:
        return None

    numeric_ref = r"(?:\d+(?:\.\d+)*|[IVXLCDM]+)"

    if _is_heading_extraction_query(clean) or _is_summary_query(clean) or _is_continuation_query(clean):
        return None

    # Concrete structural references take precedence over generic phrase parsing.
    # This catches variants such as "what is point no 3.2.2" and "heading of 3.2.2".
    single_ref = _specific_structured_reference(clean)
    if single_ref:
        kind, value = single_ref
        return {
            "type": "number",
            "number": value,
            "title": None,
            "raw": clean,
            "heading_only": kind == "heading" or _is_structured_heading_title_request(clean),
            "reference_kind": kind,
        }

    # Exclude pure conversational social queries, word count queries, or summary requests
    if re.match(r"^(?:hi|hello|hey|how\s+are\s+you|how\s+many\s+words|summariz\w*|summary|overview)\b", clean, re.I):
        return None

    # ── Category 0: Specific numbered heading references ─────────────
    # This is intentionally separate from "all headings" so a request like
    # "Tell me heading 13" returns ONLY heading 13.
    m_heading_num = re.match(
        r"^(?:tell\s+me|show\s+me|give\s+me|list|provide|display|extract|get|read|find|what\s+is|which\s+is)\s+"
        r"(?:the\s+)?heading(?:\s+(?:number|no\.?))?\s*#?\s*(" + numeric_ref + r")$",
        clean,
        re.I,
    )
    if m_heading_num and _is_specific_heading_lookup_query(clean):
        num_val = m_heading_num.group(1)
        return {"type": "number", "number": num_val, "title": None, "raw": clean, "heading_only": True}

    m_ordinal_heading = re.match(
        r"^(?:tell\s+me|show\s+me|give\s+me|list|provide|display|extract|get|read|find)\s+"
        r"(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+heading$",
        clean,
        re.I,
    )
    if m_ordinal_heading and _is_specific_heading_lookup_query(clean):
        num_val = m_ordinal_heading.group(1)
        return {"type": "number", "number": num_val, "title": None, "raw": clean, "heading_only": True}

    # ── Category 1: Section Number Queries ─────────────────────────────
    # "tell me step 5", "tell me 5", "show me section 5", "what is in step 5", "tell me 15", "give me section 15"
    m_num = re.match(
        r"^(?:tell\s+me|show\s+me|give\s+me|what\s+is\s+in|explain(?:\s+me)?|display|read|open|get|go\s+to)\s+"
        r"(?:about\s+)?(?:the\s+)?"
        r"(?:(?:step|section|subsection|sub-section|chapter|part|item|point|heading|number|no\.?|sec\.?)\s*(?:number|no\.?|#)?\s*)?"
        r"(" + numeric_ref + r")$",
        clean,
        re.I,
    )
    if m_num:
        num_val = m_num.group(1)
        return {"type": "number", "number": num_val, "title": None, "raw": clean}

    # Bare indicator + number e.g. "step 5", "section 5", "step #5", "sec 5", "no. 5"
    m_bare_ind = re.match(
        r"^(?:step|section|subsection|sub-section|chapter|part|item|point|heading|number|no\.?|sec\.?)\s*(?:number|no\.?|#)?\s*(" + numeric_ref + r")$",
        clean,
        re.I,
    )
    if m_bare_ind:
        num_val = m_bare_ind.group(1)
        return {"type": "number", "number": num_val, "title": None, "raw": clean}

    # Bare number e.g. "5" or "15"
    if re.match(rf"^({numeric_ref})$", clean):
        return {"type": "number", "number": clean, "title": None, "raw": clean}

    # ── Category 2: Explicit Content Wrapper Queries ───────────────────
    # "give me the content under/in <section>" or "tell me everything in <section>"
    m_wrap = re.search(r"\b(?:content\s+(?:under|in|of)|everything\s+in)\s+(?:the\s+)?(.+)$", clean, re.I)
    if m_wrap:
        target = _clean_section_target_text(m_wrap.group(1))
        target = re.sub(r"^(?:the\s+)?(?:section|chapter|step)\s+", "", target, flags=re.I)
        target = re.sub(r"\s+(?:section|chapter|step)$", "", target, flags=re.I).strip()
        if target:
            m_t_num = re.match(rf"^({numeric_ref})$", target)
            if m_t_num:
                return {"type": "number", "number": m_t_num.group(1), "title": None, "raw": target}
            return {"type": "title", "number": None, "title": target, "raw": target}

    # ── Category 3: Section with Suffix e.g. "tell me the X section" ───
    m_sec_suf = re.match(
        r"^(?:tell\s+me|show\s+me|give\s+me|what\s+is\s+in|explain|display|read)\s+"
        r"(?:about\s+)?(?:the\s+)?(.+?)\s+(?:section|chapter|step)$",
        clean,
        re.I,
    )
    if m_sec_suf:
        target = _clean_section_target_text(m_sec_suf.group(1))
        m_t_num = re.match(rf"^({numeric_ref})$", target)
        if m_t_num:
            return {"type": "number", "number": m_t_num.group(1), "title": None, "raw": target}
        if target and not re.search(r"\b(?:document|file|pdf|paper|image|picture)\b", target, re.I):
            return {"type": "title", "number": None, "title": target, "raw": target}

    # ── Category 4: "What is in <Section/Heading>?" ────────────────────
    m_what_in = re.match(r"^what\s+is\s+in\s+(?:the\s+)?(.+)$", clean, re.I)
    if m_what_in:
        target = _clean_section_target_text(m_what_in.group(1))
        target = re.sub(r"\s+(?:section|chapter|step)$", "", target, flags=re.I).strip()
        m_t_num = re.match(rf"^({numeric_ref})$", target)
        if m_t_num:
            return {"type": "number", "number": m_t_num.group(1), "title": None, "raw": target}
        if target and not re.search(r"\b(?:document|file|pdf|paper|image|picture)\b", target, re.I):
            return {"type": "title", "number": None, "title": target, "raw": target}

    # ── Category 5: "Tell me about / Give me / Show me <Heading>" ───────
    # e.g. "Tell me about Security and Access Rules", "Give me Conclusion and Test Notes"
    m_title = re.match(
        r"^(?:tell\s+me|show\s+me|give\s+me|explain|display|read)\s+"
        r"(?:about\s+)?(?:the\s+)?([A-Za-z0-9][\w\s&/\-–—]+)$",
        clean,
        re.I,
    )
    if m_title:
        target = _clean_section_target_text(m_title.group(1))
        # Exclude questions asking for specific personal attributes, document facts, or summaries
        if not re.search(
            r"\b(?:name|role|color|date|salary|who|why|how|candidate|applicant|intern|all\s+headings?|all\s+sections?|summary|overview|summarize|recap)\b",
            target,
            re.I,
        ):
            # Check if target begins with a section number like "5. Security and Access Rules"
            # Numeric prefixes are safe because they are unambiguous.
            # Roman numerals require a trailing dot/whitespace form; never treat
            # the first letter of a normal title (e.g. "Data Processing") as a
            # Roman-number prefix.
            m_num_title = re.match(rf"^(\d+(?:\.\d+)*)\.?\s*[:\-–—]?\s*(.+)$", target)
            if not m_num_title:
                m_num_title = re.match(r"^([IVXLCDM]{1,20})\.\s+(.+)$", target, re.IGNORECASE)
            if m_num_title:
                return {
                    "type": "number_and_title",
                    "number": m_num_title.group(1),
                    "title": m_num_title.group(2).strip(),
                    "raw": target,
                }
            m_pure_num = re.match(rf"^({numeric_ref})$", target)
            if m_pure_num:
                return {"type": "number", "number": m_pure_num.group(1), "title": None, "raw": target}
            return {"type": "title", "number": None, "title": target, "raw": target}

    return None


def _extract_target_section(question: str) -> Optional[str]:
    """Extract named section target from requests like 'tell me conclusion and test notes'."""
    parsed = _parse_section_target(question)
    if not parsed:
        return None
    if parsed.get("number") and parsed.get("title"):
        return f"{parsed['number']}. {parsed['title']}"
    return parsed.get("title") or parsed.get("number")


def _extract_document_headings(docs: list[str]) -> list[str]:
    """Deterministically extract document headings; never ask the LLM to rewrite them.

    When numbered headings are present, normalize their ordering by heading number
    rather than depending on Chroma collection iteration order. This prevents page
    markers or chunk insertion order from changing the returned table of contents.
    """
    headings = extract_structural_headings_from_chunks(docs)

    numbered = []
    unnumbered = []
    for h in headings:
        m = re.match(r"^\s*(\d+(?:\.\d+)*)\.\s+(.+?)\s*$", h)
        if m:
            number_key = tuple(int(part) for part in m.group(1).split("."))
            numbered.append((number_key, h))
        else:
            unnumbered.append(h)

    if numbered:
        numbered.sort(key=lambda item: item[0])
        result = []
        seen = set()
        for _, h in numbered:
            key = re.sub(r"\s+", " ", h.casefold()).strip()
            if key not in seen:
                seen.add(key)
                result.append(h)
        return result

    return unnumbered


def _is_summary_query(question: str) -> bool:
    """Detect document-level summary, overview, and key-points requests."""
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    q = re.sub(r"[?!.]+$", "", q).strip()

    # Specific section queries like 'what is in Executive Overview?' belong to section extraction
    if re.match(r"^what\s+is\s+in\s+", q) and not re.search(r"\b(?:this|the)?\s*(?:document|file|pdf|paper|report|text)\b", q):
        return False

    patterns = [
        r"\b(?:summariz\w*|synopsis|recap)\b",
        r"\b(?:give\s+(?:me\s+)?|provide\s+(?:me\s+)?|show\s+(?:me\s+)?|display\s+|get\s+)?(?:a\s+|an\s+)?(?:brief\s+|high-level\s+|quick\s+|general\s+)?(?:summary|overview)\b",
        r"\b(?:main\s+points?|key\s+points?|main\s+takeaways?|core\s+message|main\s+idea)\b",
        r"\bwhat\s+is\s+(?:this|the|all\s+this)?\s*(?:document|file|pdf|paper|report|text)\s+about\b",
        r"\bwhat\s+does\s+(?:this|the)\s+(?:document|file|pdf|paper|report|text)\s+(?:say|discuss|talk\s+about|cover|contain)\b",
        r"\b(?:briefly\s+explain|give\s+(?:me\s+)?(?:a\s+|an\s+)?summary|provide\s+(?:me\s+)?(?:a\s+|an\s+)?(?:summary|overview))\b",
        r"\b(?:describe|explain)\s+(?:this|the)\s+(?:document|file|pdf|paper|report)\b",
        r"\b(?:whole|complete|full|entire)\s+(?:document|file|pdf|report|paper)\b",
        r"^summary\b",
        r"^overview\b",
        r"^recap\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _has_explicit_section_cue(question: str) -> bool:
    """Return True when the wording explicitly asks for document structure.

    Generic verbs such as "tell me", "give me", and "explain" are intentionally
    not enough because their targets may be people, products, concepts, or other
    document entities rather than section titles.
    """
    q = re.sub(r"\s+", " ", (question or "").strip().casefold())
    if not q:
        return False

    explicit_patterns = (
        r"\b(?:section|sec\.?|chapter|chap\.?|part|step|item|point|heading)\b",
        r"\b(?:content|everything|details|text)\s+(?:under|inside|within)\b",
        r"\b(?:in|under|inside|within)\s+(?:the\s+)?(?:section|chapter|part|step|item|heading)\b",
        r"\b(?:section|chapter|part|step|item|heading)\s*#?\s*(?:\d+|[ivxlcdm]+)\b",
        r"\b(?:.+?)\s+(?:section|chapter|part|step)\b$",
    )
    return any(re.search(p, q, re.IGNORECASE) for p in explicit_patterns)


def classify_query_intent(
    question: str,
    image_sources: list[str],
    document_sources: list[str],
    target_source: Optional[str] = None,
) -> str:
    """Classify user query into a coarse retrieval mode.

    Section-title requests are resolved against the actual document structure by
    the service layer; this function must not assume an arbitrary noun phrase is
    a section name.
    """
    # 1. Target source points directly to an image
    if target_source and is_image_file(target_source):
        mode = "image_qa"
        print(f"[router] mode={mode} selected_source={target_source} (target is image)", file=sys.stderr)
        return mode

    # 2. Only images are uploaded in this chat -> all questions are image_qa unless explicitly asking 'define X'
    if image_sources and not document_sources:
        is_explicit_def = bool(re.search(r"\b(?:define|definition\s+of|meaning\s+of)\b", question.lower()))
        def_term = _extract_definition_term(question)
        if def_term and is_explicit_def:
            mode = "definition"
            print(f"[router] mode={mode} selected_source={target_source} (explicit def in image chat)", file=sys.stderr)
            return mode
        mode = "image_qa"
        print(f"[router] mode={mode} selected_source={target_source} (image-only chat)", file=sys.stderr)
        return mode

    # 3. Deterministic bibliography/reference queries must be document-wide and
    # cannot depend on top-k retrieval, because references may span many pages.
    if _is_reference_count_query(question):
        mode = "reference_count"
        print(f"[router] mode={mode} selected_source={target_source} (document-wide reference count)", file=sys.stderr)
        return mode
    if _is_reference_list_query(question):
        mode = "reference_list"
        print(f"[router] mode={mode} selected_source={target_source} (document-wide reference list)", file=sys.stderr)
        return mode

    # 4. Document-level summary request (MUST be evaluated before section/topic extraction)
    if _is_summary_query(question):
        mode = "summary"
        print(f"[router] mode={mode} selected_source={target_source} (summary intent)", file=sys.stderr)
        return mode

    # 5. Any single concrete structural reference must outrank global heading extraction.
    # A request like "heading of 3.2.2" is about one item, not the whole outline.
    if _is_structured_heading_title_request(question) or _is_specific_heading_lookup_query(question):
        mode = "heading_lookup"
        print(f"[router] mode={mode} selected_source={target_source} (specific structured heading lookup)", file=sys.stderr)
        return mode

    if _is_specific_structured_query(question):
        mode = "section_extraction"
        print(f"[router] mode={mode} selected_source={target_source} (specific structured reference)", file=sys.stderr)
        return mode

    # 5. Global structural heading extraction request (Part 8)
    if _is_heading_extraction_query(question):
        mode = "heading_extraction"
        print(f"[router] mode={mode} selected_source={target_source} (heading extraction)", file=sys.stderr)
        return mode

    # 6. Generic named targets are NOT enough to enter section mode. The target
    # might be a person, company, paper author, product, etc. The service layer
    # validates title-like targets against the actual document headings after it
    # has loaded the collection. Only explicit structural wording is safe to route
    # here before that document-derived validation is available.
    target_sec = _extract_target_section(question)
    if target_sec and _has_explicit_section_cue(question):
        mode = "section_extraction"
        print(f"[router] mode={mode} selected_source={target_source} (explicit section target='{target_sec}')", file=sys.stderr)
        return mode

    # 6. Any explicit document-structure reference is document-grounded.
    structured_refs = _extract_structured_references(question)
    if structured_refs:
        mode = "document_qa"
        print(
            f"[router] mode={mode} selected_source={target_source} "
            f"(structured_refs={structured_refs})",
            file=sys.stderr,
        )
        return mode

    # 7. Definition check (for document or mixed chats)
    def_term = _extract_definition_term(question)
    if def_term:
        mode = "definition"
        print(f"[router] mode={mode} selected_source={target_source} (def_term='{def_term}')", file=sys.stderr)
        return mode

    # 8. Topic check
    topic_term = _extract_topic_term(question)
    if topic_term:
        mode = "topic"
        print(f"[router] mode={mode} selected_source={target_source} (topic_term='{topic_term}')", file=sys.stderr)
        return mode

    # 9. Mixed chat: both images and documents exist
    if image_sources and document_sources:
        has_visual = _is_visual_intent(question)
        doc_mentions = bool(re.search(r"\b(?:pdf|document|documents|doc|docx|sheet|csv|resume|cv|paper|report)\b", question.lower()))

        if has_visual and not doc_mentions:
            mode = "image_qa"
            print(f"[router] mode={mode} selected_source={target_source} (mixed chat with visual intent)", file=sys.stderr)
            return mode
        if doc_mentions and not has_visual:
            mode = "document_qa"
            print(f"[router] mode={mode} selected_source={target_source} (mixed chat with document intent)", file=sys.stderr)
            return mode
        if has_visual:
            mode = "image_qa"
            print(f"[router] mode={mode} selected_source={target_source} (mixed chat visual preference)", file=sys.stderr)
            return mode

    # 10. Default document RAG
    mode = "document_qa"
    print(f"[router] mode={mode} selected_source={target_source}", file=sys.stderr)
    return mode

def _determine_query_mode(question: str, definition_mode: bool) -> str:
    """Classify the request into document_qa, definition, summary, or other."""
    if definition_mode:
        return "definition"
    q_lower = question.lower().strip()
    summary_words = {"summarize", "summary", "overview", "synopsis", "recap", "briefly describe"}
    if any(sw in q_lower for sw in summary_words):
        return "summary"
    return "document_qa"

def _safe_source_list(metas: list[dict]) -> list[str]:
    return list(dict.fromkeys(
        (m or {}).get("source", "")
        for m in metas
        if isinstance(m, dict) and (m or {}).get("source")
    ))
