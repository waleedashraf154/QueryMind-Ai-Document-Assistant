"""QueryMind FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.health import router as health_router
from backend.routes.documents import router as documents_router
from backend.routes.chat import router as chat_router
from backend.routes.chats import router as chats_router
from backend.routes.face import router as face_router
from backend.routes.transcription import router as transcription_router

app = FastAPI(
    title="QueryMind API",
    description="FastAPI backend for QueryMind AI Document Intelligence.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(chats_router)
app.include_router(face_router)
app.include_router(transcription_router)
