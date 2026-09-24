"""LLM-based intent routing for QueryMind.

The router runs before document retrieval so ordinary greetings and small talk
can be handled as normal conversation instead of being incorrectly sent through
RAG.  Classification is fail-safe: when the router cannot classify a message,
it routes to document QA so the assistant never silently answers a potentially
document-grounded question from unsupported general knowledge.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .common import _groq_client
from .generation import _call_gemini_fallback
from .tracing import _get_source_location, _langfuse, _langfuse_enabled


INTENT_ROUTER_MODEL = "openai/gpt-oss-20b"
EvidenceType = Literal["TEXT_EVIDENCE", "VISUAL_EVIDENCE", "BOTH", "GENERAL_CONVERSATION"]


class IntentClassification(BaseModel):
    """Validated result returned by the intent router."""

    evidence_type: EvidenceType = Field(
        default="TEXT_EVIDENCE",
        description=(
            "The type of evidence needed: "
            "TEXT_EVIDENCE (for fact/text queries), "
            "VISUAL_EVIDENCE (when answer depends on visual layout, spatial position, handwriting, diagrams, charts, marks), "
            "BOTH (locating text then visually inspecting layout), or "
            "GENERAL_CONVERSATION (greetings, polite chit-chat)."
        ),
    )
    intent: Literal["greeting", "casual_conversation", "document_query"] = Field(
        default="document_query",
        description=(
            "The primary intent of the user's message. "
            "Use document_query whenever the message asks for factual or document information."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)
    topic: Optional[str] = Field(
        default=None,
        description=(
            "The core entity/concept/role/term the question is about. "
            "For 'What is the color of biryani?' → 'biryani'. "
            "For 'Who is Zayn Malik?' → 'Zayn Malik'. "
            "For 'Who is the AI/ML intern?' → 'AI/ML intern'. "
            "null for social/greeting messages."
        ),
    )
    question_type: Optional[str] = Field(
        default=None,
        description=(
            "'social' for greetings/chit-chat, "
            "'document_fact' for questions whose answer should come FROM the uploaded document "
            "(e.g. a person's name in a role, an invoice total, an employee's department), "
            "'topic_knowledge' for questions about general properties/attributes of a topic "
            "(e.g. color, ingredients, history, how something works)."
        ),
    )


_INTENT_ROUTER_PROMPT = """
You are the semantic classifier, intent router, and topic extractor for QueryMind, an AI Document Assistant.

Analyze the user's message and return a JSON response with ALL six fields below.

=== FIELD 1: evidence_type ===
- GENERAL_CONVERSATION — ONLY pure social messages with ZERO information-seeking intent.
- TEXT_EVIDENCE — factual/informational question answered from text.
- VISUAL_EVIDENCE — requires visual/spatial/handwriting/diagram inspection.
- BOTH — needs text search AND visual inspection.

=== FIELD 2: intent ===
- "greeting" or "casual_conversation" — ONLY for GENERAL_CONVERSATION.
- "document_query" — for ALL information-seeking messages.

=== FIELD 3: confidence ===
Float 0.0–1.0.

=== FIELD 4: reason ===
Brief explanation (max 100 chars).

=== FIELD 5: topic ===
The core entity / concept / role / term the question is about.
Return ONLY the subject — NOT the attribute being asked about.

Examples:
"What is the color of biryani?" → topic = "biryani"   (not "color")
"What ingredients are used in biryani?" → topic = "biryani"
"Who is Zayn Malik?" → topic = "Zayn Malik"
"Who founded Pakistan?" → topic = "Pakistan"
"Who is the founder of United States?" → topic = "United States"
"What is Python?" → topic = "Python"
"Who is the AI/ML intern?" → topic = "AI/ML intern"
"Tell me the name of the AI/ML intern" → topic = "AI/ML intern"
"Who works as the AI/ML intern?" → topic = "AI/ML intern"
"What albums has Zayn Malik released?" → topic = "Zayn Malik"
"Hi" → topic = null
"How are you?" → topic = null

=== FIELD 6: question_type ===
- "social" — greeting, thanks, chit-chat, acknowledgment with no information request.
- "document_fact" — answer is a SPECIFIC FACT that exists IN the uploaded document
    (a person's name in a role, an invoice total, a date, an employee's department, an ID, etc.).
    Examples: "Who is the AI/ML intern?", "What is the invoice total?", "What is the candidate's name?"
- "topic_knowledge" — question about general properties, attributes, history, nature of a topic.
    Examples: "What is the color of biryani?", "What is Python?", "Who is Zayn Malik?",
    "What ingredients are in biryani?", "Who founded Pakistan?", "Tell me about machine learning."

=== CRITICAL RULES ===
1. GENERAL_CONVERSATION = ONLY pure social messages: hi, hello, thanks, bye, ok, cool, good morning.
   NEVER classify information-seeking questions as GENERAL_CONVERSATION.
2. "Who is X?", "What is X?", "Tell me about X", "Explain X" — ALWAYS document_query.
3. topic: extract the SUBJECT, not the attribute.
   "color of biryani" → "biryani". "founder of Pakistan" → "Pakistan". "AI/ML intern" → "AI/ML intern".
4. question_type document_fact = the specific answer lives IN the uploaded document.
   question_type topic_knowledge = general knowledge about the topic (may or may not be in doc).
5. When in doubt between GENERAL_CONVERSATION and document_query → choose document_query.

Return ONLY valid JSON with exactly these keys:
{
  "evidence_type": "TEXT_EVIDENCE" | "VISUAL_EVIDENCE" | "BOTH" | "GENERAL_CONVERSATION",
  "intent": "greeting" | "casual_conversation" | "document_query",
  "confidence": 0.0,
  "reason": "...",
  "topic": "extracted topic or entity" | null,
  "question_type": "social" | "document_fact" | "topic_knowledge"
}
""".strip()


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(raw: str) -> str:
    """Extract the first JSON object if the model wrapped it in prose."""
    text = _strip_code_fences(raw)
    if not text:
        raise ValueError("Intent router returned an empty response.")

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Intent router did not return a JSON object.")

    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate


def _parse_classification(raw: str) -> IntentClassification:
    """Normalize and validate model output through the Pydantic schema."""
    json_text = _extract_json_object(raw)
    try:
        payload = json.loads(json_text)
        if not isinstance(payload, dict):
            raise ValueError("Intent router returned a JSON value that is not an object.")

        evidence_raw = str(payload.get("evidence_type", "")).strip().upper()
        if evidence_raw in ("TEXT_EVIDENCE", "VISUAL_EVIDENCE", "BOTH", "GENERAL_CONVERSATION"):
            evidence_type = evidence_raw
        else:
            intent_raw = str(payload.get("intent", "")).strip().lower()
            if intent_raw in ("greeting", "casual_conversation"):
                evidence_type = "GENERAL_CONVERSATION"
            else:
                evidence_type = "TEXT_EVIDENCE"
        payload["evidence_type"] = evidence_type

        # Synchronize intent
        if evidence_type == "GENERAL_CONVERSATION":
            intent = str(payload.get("intent", "")).strip().lower()
            intent_aliases = {
                "chat": "casual_conversation",
                "general_chat": "casual_conversation",
                "small_talk": "casual_conversation",
                "smalltalk": "casual_conversation",
                "conversation": "casual_conversation",
                "greeting_or_chat": "greeting",
            }
            payload["intent"] = intent_aliases.get(intent, "greeting" if "greet" in intent else "casual_conversation")
        else:
            payload["intent"] = "document_query"

        # Normalize topic — treat empty strings / "null" / "none" as None
        raw_topic = payload.get("topic", None)
        if isinstance(raw_topic, str):
            raw_topic = raw_topic.strip()
            if not raw_topic or raw_topic.lower() in ("null", "none", "n/a", ""):
                raw_topic = None
        payload["topic"] = raw_topic or None

        # Normalize question_type — allow only the three valid values
        raw_qtype = str(payload.get("question_type", "")).strip().lower()
        if raw_qtype not in ("social", "document_fact", "topic_knowledge"):
            # Infer from evidence_type when model doesn't return the new field
            raw_qtype = "social" if evidence_type == "GENERAL_CONVERSATION" else None
        payload["question_type"] = raw_qtype or None

        return IntentClassification.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"Invalid intent-router output: {exc}") from exc


def _get_router_system_prompt() -> str:
    try:
        from backend.core.langfuse import get_managed_prompt
        prompt_text, _ = get_managed_prompt("routing/querymind-intent-router")
        return prompt_text
    except Exception:
        return _INTENT_ROUTER_PROMPT


def _groq_classify(question: str):
    if _groq_client is None:
        raise RuntimeError("Groq client is not configured.")

    prompt_content = _get_router_system_prompt()
    messages = [
        {"role": "system", "content": prompt_content},
        {"role": "user", "content": question.strip()},
    ]

    kwargs = {
        "model": INTENT_ROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": 600,  # increased to accommodate topic + question_type fields
    }

    # Prefer JSON mode, but retry without response_format for compatibility
    # with providers/models that do not expose that parameter.
    try:
        return _groq_client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as first_error:
        print(
            f"[intent-router] Groq JSON mode failed; retrying without response_format: {first_error}",
            file=sys.stderr,
        )
        return _groq_client.chat.completions.create(**kwargs)


def _gemini_classify(question: str):
    return _call_gemini_fallback(_get_router_system_prompt(), question.strip())


def _record_router_trace(
    question: str,
    result: IntentClassification,
    *,
    model: str,
    latency_ms: float,
    root_obs=None,
) -> None:
    """Record the router decision in Langfuse when tracing is enabled."""
    if not _langfuse_enabled() or _langfuse is None:
        if root_obs is not None:
            root_obs.update(
                metadata={
                    "intent": result.intent,
                    "intent_confidence": result.confidence,
                }
            )
        return

    source_info = _get_source_location(classify_intent)
    with _langfuse.start_as_current_observation(
        as_type="chain",
        name="intent-router",
        input={"question": question},
        metadata={
            "model": model,
            "intent": result.intent,
            "confidence": result.confidence,
            "latency_ms": latency_ms,
            **source_info,
        },
    ) as observation:
        observation.update(
            output={
                "intent": result.intent,
                "confidence": result.confidence,
                "reason": result.reason,
            },
            metadata={
                "model": model,
                "latency_ms": latency_ms,
            },
        )

    if root_obs is not None:
        root_obs.update(
            metadata={
                "intent": result.intent,
                "intent_confidence": result.confidence,
            }
        )


def _safe_document_fallback(question: str, reason: str) -> IntentClassification:
    print(
        f"[intent-router] Falling back to document_query for safety: {reason}",
        file=sys.stderr,
    )
    return IntentClassification(
        evidence_type="TEXT_EVIDENCE",
        intent="document_query",
        confidence=0.0,
        reason=f"Routing fallback: document-grounded handling is the safe path ({reason}).",
    )


def classify_intent(
    question: str,
    *,
    selected_source: Optional[str] = None,
    root_obs=None,
) -> IntentClassification:
    """Classify a message using semantic LLM intent routing with safe RAG fallback."""
    clean_question = (question or "").strip()
    if not clean_question:
        return IntentClassification(
            evidence_type="GENERAL_CONVERSATION",
            intent="casual_conversation",
            confidence=1.0,
            reason="Empty input is treated as conversational rather than document retrieval.",
        )

    started = time.perf_counter()
    result: Optional[IntentClassification] = None
    used_model = INTENT_ROUTER_MODEL

    try:
        response = _groq_classify(clean_question)
        raw = response.choices[0].message.content or ""
        result = _parse_classification(raw)
    except Exception as groq_exc:
        print(f"[intent-router] Groq classification failed: {groq_exc}", file=sys.stderr)
        try:
            response = _gemini_classify(clean_question)
            raw = response.choices[0].message.content or ""
            result = _parse_classification(raw)
            used_model = "google/gemini-fallback"
        except Exception as gemini_exc:
            result = _safe_document_fallback(clean_question, str(gemini_exc))

    # If the user explicitly focused on a specific document source and the
    # intent was classified as casual_conversation but contains a substantive question/topic,
    # route to document_query for safety.
    if selected_source and result.evidence_type == "GENERAL_CONVERSATION":
        if "?" in clean_question or len(clean_question.split()) > 5:
            result = IntentClassification(
                evidence_type="TEXT_EVIDENCE",
                intent="document_query",
                confidence=0.85,
                reason=f"Selected source '{selected_source}' active with inquiry query.",
            )

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    _record_router_trace(
        clean_question,
        result,
        model=used_model,
        latency_ms=latency_ms,
        root_obs=root_obs,
    )
    print(
        f"[intent-router] evidence_type={result.evidence_type} intent={result.intent} "
        f"confidence={result.confidence:.2f} model={used_model} latency_ms={latency_ms}",
        file=sys.stderr,
    )
    return result


# Maintain backward compatibility for any callers expecting classify_user_message
classify_user_message = classify_intent
