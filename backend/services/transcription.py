"""Audio transcription application service."""
import os
import sys

from backend.core.langfuse import get_managed_prompt

async def transcribe_audio(file) -> dict:
    content = await file.read()
    if not content:
        raise ValueError("Empty audio file provided.")

    filename = file.filename or "audio.webm"
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            res = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(filename, content),
                language="en",
                response_format="json",
            )
            return {"text": (res.text or "").strip()}
        except Exception as groq_err:
            print(f"[transcribe] Groq transcription error: {groq_err}")

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini_key)
            mime = file.content_type or "audio/webm"
            instruction, _prompt = get_managed_prompt("transcription/querymind-transcription")
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[
                    types.Part.from_bytes(data=content, mime_type=mime),
                    instruction,
                ],
            )
            return {"text": (response.text or "").strip()}
        except Exception as gemini_err:
            print(f"[transcribe] Gemini transcription error: {gemini_err}")

    raise RuntimeError("No transcription service configured or available.")
