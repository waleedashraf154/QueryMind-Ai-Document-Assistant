"""Shared backend configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

STORAGE_DIR = BACKEND_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".pptx", ".xlsx", ".csv",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
}
