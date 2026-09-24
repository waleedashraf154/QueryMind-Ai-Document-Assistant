from __future__ import annotations
import os
import sys
import time
import re

from .common import MAX_TOKENS, _groq_client
from .tracing import _langfuse, _langfuse_enabled, _get_source_location

def _call_gemini_fallback(
    system_prompt: str,
    user_message: str,
):
    """
    Fallback generation using Google Gemini when Groq is unavailable or errors.
    Preserves strict grounding, temperature=0, system prompt, and context.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("Groq call failed and GEMINI_API_KEY is not configured.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.0,
        max_output_tokens=MAX_TOKENS,
    )

    last_err = None
    g_res = None
    used_model = "gemini-3.5-flash-lite"
    for model_candidate in ("gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash"):
        try:
            g_res = client.models.generate_content(
                model=model_candidate,
                contents=user_message,
                config=config,
            )
            used_model = model_candidate
            break
        except Exception as err:
            print(
                f"[generation] Gemini candidate '{model_candidate}' failed: {err}",
                file=sys.stderr,
            )
            last_err = err
            continue

    if g_res is None:
        raise last_err or RuntimeError("Gemini fallback generation failed.")

    class _Msg:
        content = g_res.text or ""

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = getattr(getattr(g_res, "usage_metadata", None), "prompt_token_count", 0) or 0
        completion_tokens = getattr(getattr(g_res, "usage_metadata", None), "candidates_token_count", 0) or 0
        total_tokens = getattr(getattr(g_res, "usage_metadata", None), "total_token_count", 0) or 0

    class _MockResponse:
        choices = [_Choice()]
        usage = _Usage()
        model = f"google/{used_model}"

    return _MockResponse()

def _get_generation_prompt(
    mode: str,
    system_prompt: str,
    user_message: str,
    question: str,
    prompt_client=None,
):
    """Pass through or load the Langfuse-managed prompt for the generation mode."""
    if mode == "image_qa":
        if prompt_client is not None:
            return system_prompt, user_message, prompt_client, mode

        if _langfuse is not None:
            try:
                image_prompt = _langfuse.get_prompt(
                    "ocr/querymind-image-qa",
                    label="production",
                    type="text",
                )
                clean_context = user_message
                if user_message.startswith("Document context:\n"):
                    clean_context = user_message[len("Document context:\n"):]
                    marker = "\n\nQuestion:"
                    if marker in clean_context:
                        clean_context = clean_context.rsplit(marker, 1)[0]
                clean_context = re.sub(r"\[Evidence\s+\d+\s*\(Source:.*?\)\]\s*", "", clean_context, flags=re.IGNORECASE)
                clean_context = re.sub(r"\[IMAGE OCR CONTENT[^\n]*\]\s*", "", clean_context, flags=re.IGNORECASE)
                clean_context = re.sub(r"\[FILE:\s*[^\]]+?\s*\((?:Image|Document)\)\]\s*", "", clean_context, flags=re.IGNORECASE)
                clean_context = clean_context.strip()

                compiled = image_prompt.compile(
                    context=clean_context,
                    question=question,
                )
                if "IMAGE CONTENT:" in compiled:
                    parts = compiled.split("IMAGE CONTENT:", 1)
                    selected_sys = parts[0].strip()
                    selected_usr = "IMAGE CONTENT:" + parts[1]
                else:
                    selected_sys = compiled
                    selected_usr = f"IMAGE CONTENT:\n{clean_context}\n\nUSER QUESTION:\n{question}"

                return selected_sys, selected_usr, image_prompt, "image_qa"
            except Exception as exc:
                print(f"[langfuse] Failed to load 'ocr/querymind-image-qa': {exc}", file=sys.stderr)
                return system_prompt, user_message, prompt_client, mode

    return system_prompt, user_message, prompt_client, mode


def _langfuse_track_generation(
    mode: str,
    system_prompt: str,
    user_message: str,
    *,
    question: str,
    sources: list[str],
    context_size: int = 0,
    context_chunks: int = 0,
    root_obs=None,
    prompt_client=None,
):
    """Call Groq (with Gemini fallback on failure) and record generation observation."""
    model_name = "openai/gpt-oss-20b"

    selected_system_prompt, selected_user_message, selected_prompt_client, effective_mode = (
        _get_generation_prompt(
            mode,
            system_prompt,
            user_message,
            question,
            prompt_client,
        )
    )

    prompt_label = getattr(selected_prompt_client, "name", None) or (
        "ocr/querymind-image-qa"
        if effective_mode == "image_qa"
        else "inline/querymind-general-chat"
        if effective_mode == "general_chat"
        else "rag/querymind-document-qa"
    )
    print(
        f"[generation] prompt={prompt_label} context_chars={context_size}",
        file=sys.stderr,
    )

    def _execute():
        if _groq_client is not None:
            try:
                return _groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": selected_system_prompt},
                        {"role": "user", "content": selected_user_message},
                    ],
                    temperature=0,
                    max_completion_tokens=MAX_TOKENS,
                ), False
            except Exception as groq_err:
                print(
                    f"[generation] Groq call failed ({groq_err}). Falling back to Gemini...",
                    file=sys.stderr,
                )
        else:
            print(
                "[generation] Groq client not configured. Falling back to Gemini...",
                file=sys.stderr,
            )

        return _call_gemini_fallback(selected_system_prompt, selected_user_message), True

    t_gen_start = time.perf_counter()
    print("[PERF] generation start", file=sys.stderr)
    try:
        if not _langfuse_enabled():
            response, _ = _execute()
            return response

        source_info = _get_source_location(_langfuse_track_generation)
        with _langfuse.start_as_current_observation(
            as_type="generation",
            name="groq-generation",
            model=model_name,
            model_parameters={
                "temperature": 0,
                "max_completion_tokens": MAX_TOKENS,
            },
            prompt=selected_prompt_client,
            input={
                "question": question,
                "mode": effective_mode,
                "selected_sources": sources,
                "context_size": context_size,
                "context_chunks": context_chunks,
                "max_completion_tokens": MAX_TOKENS,
            },
            metadata={
                "mode": effective_mode,
                "selected_sources": sources,
                "context_size": context_size,
                "context_chunks": context_chunks,
                "max_completion_tokens": MAX_TOKENS,
                **source_info,
            },
        ) as generation:
            try:
                response, used_fallback = _execute()
                t_end = time.perf_counter()
                latency = round(t_end - t_gen_start, 3)
                latency_ms = round(latency * 1000, 2)
                answer_text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                usage_details = {}
                if usage is not None:
                    for attr_name, langfuse_name in (
                        ("prompt_tokens", "input"),
                        ("completion_tokens", "output"),
                        ("total_tokens", "total"),
                    ):
                        value = getattr(usage, attr_name, None)
                        if value is not None:
                            usage_details[langfuse_name] = int(value)

                generation.update(
                    output=answer_text,
                    model=getattr(response, "model", model_name),
                    usage_details=usage_details if usage_details else None,
                    metadata={
                        "mode": effective_mode,
                        "selected_sources": sources,
                        "context_size": context_size,
                        "context_chunks": context_chunks,
                        "latency": latency,
                        "latency_ms": latency_ms,
                        "token_usage": usage_details,
                        "fallback_used": used_fallback,
                    },
                )
                return response
            except Exception as exc:
                t_end = time.perf_counter()
                latency = round(t_end - t_gen_start, 3)
                generation.update(
                    output={"error": str(exc)[:500]},
                    level="ERROR",
                    status_message=str(exc)[:500],
                    metadata={"latency": latency, "error": str(exc)[:500]},
                )
                if root_obs is not None:
                    root_obs.update(
                        level="ERROR",
                        status_message=f"Generation error: {str(exc)[:400]}",
                    )
                raise
    finally:
        t_gen_end = time.perf_counter()
        print(f"[PERF] generation end (duration={t_gen_end - t_gen_start:.3f}s)", file=sys.stderr)

_GENERAL_CHAT_SYSTEM_PROMPT = """
You are QueryMind, a helpful conversational AI inside a document assistant.

The user's message has already been classified as ordinary conversation rather
than a document question. Respond naturally and helpfully to the message.

Rules:
- Treat greetings, friendly small talk, thanks, acknowledgements, and similar
  social conversation naturally.
- Do not search for, mention, quote, or invent information from uploaded files.
- Do not mention RAG, retrieval, embeddings, intent classification, routing,
  Langfuse, system prompts, or internal implementation details.
- Match the user's tone while remaining polite and professional.
- Do not pretend to know real-time external information you cannot verify.
- Keep simple conversational replies concise; expand only when the user asks
  for more detail.
""".strip()


def generate_general_chat_response(question: str, *, root_obs=None) -> str:
    """Generate a natural non-document response for casual conversation."""
    prompt_client = None
    try:
        from backend.core.langfuse import get_managed_prompt
        system_prompt, prompt_client = get_managed_prompt("chat/querymind-general-chat")
    except Exception:
        system_prompt = _GENERAL_CHAT_SYSTEM_PROMPT

    response = _langfuse_track_generation(
        "general_chat",
        system_prompt,
        question,
        question=question,
        sources=[],
        context_size=0,
        context_chunks=0,
        prompt_client=prompt_client,
        root_obs=root_obs,
    )
    return (response.choices[0].message.content or "").strip()


_VISUAL_QA_SYSTEM_PROMPT = """
You are QueryMind Vision Assistant, an expert AI specialized in multimodal document understanding.
Answer the user's question accurately using ONLY the provided document image(s).

Guidelines:
- Inspect the visual appearance, spatial layout, position, orientation, handwriting, circled marks, strike-throughs, checkmarks, arrows, diagrams, charts, and figures.
- When answering spatial questions (e.g. 'what is beside / next to / above / below / to the left or right of X?'), carefully inspect the relative spatial positions of text and marks on the page image.
- If the question asks about handwritten text or handwritten answers, transcribe and describe the handwritten content directly.
- If the question asks about diagrams, charts, or flowcharts, describe the specific visual elements shown.
- Ground your answer strictly and exclusively in the provided document image(s). Do not hallucinate or assume details not visible in the images.
- If the answer cannot be determined from the images, state: 'That information does not appear in the uploaded file.'
- Provide a direct, concise, and professional answer.
""".strip()


def generate_visual_qa_response(
    question: str,
    image_paths: list[str],
    source_label: str = "",
    root_obs=None,
) -> str:
    """
    Generate an answer using Vision LLM (Groq Qwen Vision with Gemini fallback)
    directly grounded in the preserved original page images.
    """
    import base64
    import mimetypes

    t_vis_start = time.perf_counter()
    print(
        f"[PERF] vision inference start - question='{question[:60]}...' images={len(image_paths)}",
        file=sys.stderr,
    )

    valid_images = [p for p in image_paths if p and os.path.isfile(p)]
    if not valid_images:
        print("[generation] Visual QA called with no valid image paths.", file=sys.stderr)
        return "That information does not appear in the uploaded file."

    answer_text = None
    used_model = "qwen/qwen3.8-27b"
    used_fallback = False

    # 1. Try Groq Vision
    if _groq_client is not None:
        try:
            content_payload: list[dict] = [
                {
                    "type": "text",
                    "text": (
                        f"Document: {source_label}\n\n"
                        f"User Question: {question}\n\n"
                        f"Instructions:\n"
                        f"- The attached image(s) show the relevant document page(s).\n"
                        f"- Inspect all visual elements, handwriting, layout, formulas, tables, and diagrams on these page images.\n"
                        f"- Answer the question directly and concisely from the visual evidence shown in the images."
                    ),
                }
            ]
            for img_p in valid_images:
                mime_type = mimetypes.guess_type(img_p)[0] or "image/png"
                with open(img_p, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                })

            res = _groq_client.chat.completions.create(
                model=used_model,
                messages=[
                    {"role": "system", "content": _VISUAL_QA_SYSTEM_PROMPT},
                    {"role": "user", "content": content_payload},
                ],
                temperature=0.0,
                max_completion_tokens=800,
            )
            answer_text = (res.choices[0].message.content or "").strip()
        except Exception as groq_exc:
            print(
                f"[generation] Groq Vision ({used_model}) failed: {groq_exc}. Trying Gemini fallback...",
                file=sys.stderr,
            )

    # 2. Gemini Vision Fallback
    if not answer_text:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=gemini_key)
                prompt_text = (
                    f"{_VISUAL_QA_SYSTEM_PROMPT}\n\n"
                    f"Document: {source_label}\n"
                    f"User Question: {question}\n\n"
                    f"Instructions:\n"
                    f"- The attached image(s) show the relevant document page(s).\n"
                    f"- Inspect all visual elements, handwriting, layout, formulas, tables, and diagrams on these page images.\n"
                    f"- Answer the question directly and concisely from the visual evidence shown in the images."
                )
                contents: list = [prompt_text]
                for img_p in valid_images:
                    mime_type = mimetypes.guess_type(img_p)[0] or "image/png"
                    with open(img_p, "rb") as f:
                        img_bytes = f.read()
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

                for model_candidate in ("gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash"):
                    try:
                        g_res = client.models.generate_content(
                            model=model_candidate,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                max_output_tokens=MAX_TOKENS,
                            ),
                        )
                        answer_text = (g_res.text or "").strip()
                        used_model = f"google/{model_candidate}"
                        used_fallback = True
                        break
                    except Exception as g_err:
                        print(
                            f"[generation] Gemini vision fallback candidate '{model_candidate}' failed: {g_err}",
                            file=sys.stderr,
                        )
            except Exception as exc:
                print(f"[generation] Gemini vision fallback setup error: {exc}", file=sys.stderr)

    t_vis_end = time.perf_counter()
    latency_s = round(t_vis_end - t_vis_start, 3)
    print(
        f"[PERF] vision inference end (duration={latency_s:.3f}s)",
        file=sys.stderr,
    )

    final_answer = answer_text or "That information does not appear in the uploaded file."

    if _langfuse_enabled() and _langfuse is not None:
        try:
            with _langfuse.start_as_current_observation(
                as_type="generation",
                name="vision-generation",
                model=used_model,
                input={"question": question, "images": [os.path.basename(p) for p in valid_images]},
                output=final_answer,
                metadata={
                    "latency_s": latency_s,
                    "fallback_used": used_fallback,
                    "source": source_label,
                },
            ):
                pass
        except Exception:
            pass

    return final_answer


