"""Chat application service."""
import sys
import backend.database.storage as storage
from backend.rag import answer_question
from backend.services.title_generator import generate_chat_title

def answer_chat(chat_id: str, user_email: str, question: str, selected_source: str | None = None) -> dict:
    storage.require_chat_owner(chat_id, user_email)
    storage.add_message(chat_id, "user", question)
    conversation_history = storage.get_recent_messages(chat_id, limit=12)

    new_chat_title: str | None = None

    # Step 1: One-time automatic title generation
    # - Only runs if title_generated is still FALSE.
    # - Generates a concise topic headline from the very FIRST user message.
    # - Atomic update prevents race conditions if multiple requests arrive close together.
    # - Failures must never break the chat answering flow.
    try:
        if not storage.is_chat_title_generated(chat_id):
            first_user_msg = storage.get_first_user_message(chat_id) or question
            generated_title = generate_chat_title(first_user_msg)
            if generated_title and generated_title not in ("New Chat", "Untitled chat"):
                updated = storage.update_chat_title_if_not_generated(chat_id, generated_title)
                if updated:
                    new_chat_title = generated_title
    except Exception as title_err:
        print(f"[chat] Non-fatal automatic title generation error: {title_err}", file=sys.stderr)

    # Step 2: Generate normal answer via RAG pipeline
    try:
        result = answer_question(
            chat_id,
            question,
            selected_source=selected_source,
            user_email=user_email,
            conversation_history=conversation_history,
        )
        if isinstance(result, dict):
            answer = result.get("answer", "")
            sources = result.get("sources", [])
        else:
            answer = str(result)
            sources = []
    except Exception as exc:
        print(f"[chat] Chat processing error: {exc}", file=sys.stderr)
        answer = (
            "⚠️ An unexpected error occurred while processing your question. "
            "Please try again or rephrase your query."
        )
        sources = []

    # Step 3: Record assistant response
    storage.add_message(chat_id, "assistant", answer)

    response_payload = {
        "answer": answer,
        "sources": sources,
    }
    if new_chat_title:
        response_payload["chat_title"] = new_chat_title

    return response_payload
