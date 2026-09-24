"""Face authentication application services."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from backend.services.face_engine import register_face, match_face
from backend.services.auth import has_face_registered, delete_face_embedding

FACE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face-ml")

async def register(email: str, image_data: str, overwrite: bool = False) -> dict:
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(
            FACE_EXECUTOR,
            lambda: register_face(email, image_data, overwrite=overwrite),
        ),
        timeout=25.0,
    )

def match(image_data: str) -> dict:
    return match_face(image_data)

def status(email: str) -> dict:
    return {"registered": has_face_registered(email.strip().lower())}

def delete(email: str) -> dict:
    ok, message = delete_face_embedding(email.strip().lower())
    if not ok:
        raise FileNotFoundError(message)
    return {"status": "ok", "message": message}
