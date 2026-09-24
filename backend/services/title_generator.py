"""Lightweight one-time chat title generation service.

Generates concise 1-5 word topic headlines from the first user message.
Reuses existing Groq (openai/gpt-oss-20b) and Gemini fallback infrastructure
without touching Chroma retrieval, embeddings, or the document RAG pipeline.
"""
import os
import re
import sys
from typing import Optional

# Common casual/greeting phrases mapped directly to clean titles without LLM calls
GREETING_PATTERNS = {
    "hi", "hello", "hey", "how are you", "how are you?", "what's up",
    "whats up", "what's up?", "good morning", "good afternoon", "good evening",
    "greetings", "salut", "hola", "yo", "hey there", "hello there"
}

TITLE_SYSTEM_PROMPT = (
    "You generate a concise title for a chat based on the user's first message.\n\n"
    "Rules:\n"
    "- Return ONLY the title.\n"
    "- 1 to 5 words preferably.\n"
    "- Make it a topic/headline, not a sentence.\n"
    "- Capture the main subject of the user's message.\n"
    "- Do not include quotation marks.\n"
    "- Do not include prefixes such as 'Title:'.\n"
    "- Do not end with punctuation.\n"
    "- Do not mention ChatGPT, assistant, conversation, or user.\n"
    "- Do not answer the user's question.\n"
    "- Do not explain anything.\n"
    "- Preserve important proper nouns, abbreviations, and technical terms.\n"
    "- Make the title natural and readable.\n\n"
    "Examples:\n"
    'User: "What is education?" -> Education\n'
    'User: "Explain machine learning" -> Machine Learning\n'
    'User: "What is a Pydantic model?" -> Pydantic Models\n'
    'User: "How does OCR work?" -> OCR\n'
    'User: "Tell me about Pakistan history" -> Pakistan History\n'
    'User: "What are the benefits of exercise?" -> Exercise Benefits\n'
)

_groq_client_instance = None

def _get_groq_client():
    """Lazily instantiate Groq client to avoid top-level dependencies."""
    global _groq_client_instance
    if _groq_client_instance is not None:
        return _groq_client_instance
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        _groq_client_instance = Groq(api_key=api_key)
        return _groq_client_instance
    except Exception as err:
        print(f"[title_generator] Failed to initialize Groq client: {err}", file=sys.stderr)
        return None


def clean_title(raw_title: str) -> str:
    """Sanitize the raw model response into a clean 1-5 word headline."""
    if not raw_title:
        return "New Chat"

    # Take first non-empty line
    lines = [line.strip() for line in raw_title.strip().splitlines() if line.strip()]
    if not lines:
        return "New Chat"
    cleaned = lines[0]

    # Strip prefixes like "Title:", "Headline:", "Topic:", etc.
    cleaned = re.sub(r"^(?:Title|Headline|Subject|Topic|Chat Title)\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    # Strip wrapping quotes, backticks, brackets
    cleaned = cleaned.strip("\"'“”`«»[]()")

    # Strip trailing punctuation (periods, commas, exclamation marks, question marks, colons)
    cleaned = re.sub(r"[\.\,\!\?\:]+$", "", cleaned).strip()

    # Limit length to ~60 characters to ensure concise headline
    if len(cleaned) > 60:
        cleaned = cleaned[:60].rsplit(" ", 1)[0].strip()

    return cleaned or "New Chat"


def generate_chat_title(first_user_message: str) -> str:
    """
    Generate a concise topic title based on the user's first message.

    - Title is generated only once per chat.
    - Casual greetings return a friendly topic title immediately without LLM calls.
    - Uses Groq (openai/gpt-oss-20b) with Gemini fallback on failure.
    - Title-generation failures must never break chat answering (caller catches errors).
    """
    clean_msg = (first_user_message or "").strip()
    if not clean_msg:
        return "New Chat"

    # Fast-path for simple casual conversation / greetings
    normalized_greeting = clean_msg.lower().strip("!?., ")
    if normalized_greeting in GREETING_PATTERNS:
        return "Greetings"

    model_name = "openai/gpt-oss-20b"
    prompt_messages = [
        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": f'User: "{clean_msg[:400]}"'},
    ]

    raw_text = ""
    client = _get_groq_client()
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=prompt_messages,
                temperature=0,
                max_completion_tokens=200,
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception as groq_err:
            print(f"[title_generator] Groq call failed ({groq_err}), attempting Gemini fallback...", file=sys.stderr)

    # Gemini fallback
    if not raw_text:
        try:
            from backend.rag_pipeline.generation import _call_gemini_fallback
            resp = _call_gemini_fallback(
                system_prompt=TITLE_SYSTEM_PROMPT,
                user_message=f'User: "{clean_msg[:400]}"',
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception as gemini_err:
            print(f"[title_generator] Gemini fallback also failed: {gemini_err}", file=sys.stderr)

    return clean_title(raw_text)
