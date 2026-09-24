"""
backend/parser.py
─────────────────
extract_text(file_path: str) -> str

Extracts all readable text from a document file.
Supported extensions (matching Frontend's st.chat_input file_type list):
    .pdf    → PyMuPDF/pypdf, with Vision OCR fallback for scanned/image PDFs
    .docx   → python-docx, plus Vision OCR for embedded images
    .pptx   → python-pptx, plus Vision OCR for embedded images
    .xlsx   → pandas + openpyxl
    .csv    → pandas
    .txt    → plain read

Raises ValueError for unsupported file types.
"""

import os
import sys
import re
import time
import mimetypes
import zipfile
from typing import Optional


from backend.core.config import BACKEND_DIR
from backend.core.langfuse import get_managed_prompt as _fetch_managed_prompt


def _get_managed_prompt(name: str, **variables) -> str:
    """Fetch and compile a production prompt managed in Langfuse."""
    return _fetch_managed_prompt(name, **variables)[0]


# Number of data rows to group into a single text chunk for tabular files.
# Keeping this at ~25 rows means a 700-row CSV produces ≈28 chunks instead
# of ≈700, staying comfortably within Gemini's free-tier embedding quota.
TABULAR_ROWS_PER_CHUNK: int = 25


# ──────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """
    Extract all text from *file_path* and return it as a single string.

    Parameters
    ----------
    file_path : str
        Absolute (or relative) path to the document file.

    Returns
    -------
    str
        The full extracted text. May be empty if the document has no
        readable content, but will never be None.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".pptx":
        return _extract_pptx(file_path)
    elif ext == ".xlsx":
        return _extract_xlsx(file_path)
    elif ext == ".csv":
        return _extract_csv(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return _extract_image(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            "Supported types: pdf, docx, pptx, xlsx, csv, txt, png, jpg, jpeg, webp, bmp."
        )


# ──────────────────────────────────────────────────────────────
# PRIVATE EXTRACTORS
# ──────────────────────────────────────────────────────────────

import time
from pathlib import Path


def _gemini_vision_fallback(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    source_label: str,
) -> str:
    """Fallback vision extraction using Google Gemini when Groq Vision fails or hits rate limits."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set for Gemini Vision fallback.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_key)
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    candidate_models = (
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.5-flash",
    )

    for model_name in candidate_models:
        for attempt in range(3):
            try:
                print(
                    f"[parser] Gemini Vision fallback: model={model_name}, attempt={attempt+1}, source={source_label}",
                    file=sys.stderr,
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=[part, prompt],
                )
                text = (response.text or "").strip()
                if text:
                    return text
                break
            except Exception as err:
                err_msg = str(err)
                print(
                    f"[parser] Gemini Vision candidate '{model_name}' attempt {attempt+1} failed for {source_label}: {err_msg[:160]}",
                    file=sys.stderr,
                )
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or 
                    "503" in err_msg or "UNAVAILABLE" in err_msg or "high demand" in err_msg):
                    delay = 2.0 * (attempt + 1)
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                    if m:
                        delay = max(delay, float(m.group(1)) + 0.5)
                    delay = min(delay, 12.0)
                    print(f"[parser] Waiting {delay:.1f}s before retrying {model_name}...", file=sys.stderr)
                    time.sleep(delay)
                else:
                    break
    raise RuntimeError(f"All Gemini Vision candidates failed for {source_label}.")


# ── Groq Vision 429 Circuit Breaker ─────────────────────────
_groq_vision_blocked_until: float = 0.0
_GROQ_COOLDOWN_SECONDS: float = 300.0  # 5 minutes


def reset_groq_vision_cooldown() -> None:
    """Reset the Groq Vision circuit breaker (useful for testing or manual retry)."""
    global _groq_vision_blocked_until
    _groq_vision_blocked_until = 0.0


def _extract_vision_from_bytes(
    image_bytes: bytes,
    mime_type: str,
    source_label: str,
    *,
    max_completion_tokens: int = 800,
) -> str:
    """
    Extract text and visual information from image bytes using Groq Vision
    (qwen/qwen3.8-27b) with Gemini Vision fallback.

    This helper is shared between standalone image parsing and images
    embedded inside Office documents. Keeping one implementation ensures all
    visual inputs use the same Vision prompt and robust multi-provider fallback.
    """
    global _groq_vision_blocked_until
    import base64
    from groq import Groq  # type: ignore

    prompt = _get_managed_prompt("ocr/querymind-vision")
    groq_key = os.getenv("GROQ_API_KEY")

    now = time.time()
    if groq_key and now < _groq_vision_blocked_until:
        remaining = int(_groq_vision_blocked_until - now)
        print(
            f"[parser] Groq Vision in cooldown (429 circuit breaker active, {remaining}s remaining). "
            f"Skipping Groq, using Gemini directly for {source_label}.",
            file=sys.stderr,
        )
    elif groq_key:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{base64_image}"
        client = Groq(api_key=groq_key)
        model_name = "qwen/qwen3.8-27b"

        print(
            f"[parser] Groq Vision: model={model_name}, source={source_label}",
            file=sys.stderr,
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                temperature=0,
                max_completion_tokens=max_completion_tokens,
                stream=False,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            print(
                f"[parser] Groq model '{model_name}' returned empty text for {source_label}. Trying Gemini fallback...",
                file=sys.stderr,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                _groq_vision_blocked_until = time.time() + _GROQ_COOLDOWN_SECONDS
                print(
                    f"[parser] Groq Vision 429 rate limit reached ({exc}). "
                    f"Activating circuit breaker for {_GROQ_COOLDOWN_SECONDS}s. "
                    f"Remaining pages will use Gemini Vision directly.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[parser] Groq Vision failed: model={model_name}, source={source_label}, error={exc}. Trying Gemini fallback...",
                    file=sys.stderr,
                )

    # Gemini Vision fallback
    try:
        return _gemini_vision_fallback(image_bytes, mime_type, prompt, source_label)
    except Exception as fallback_exc:
        print(
            f"[parser] Vision analysis failed on both Groq and Gemini for {source_label}: {fallback_exc}",
            file=sys.stderr,
        )
        raise RuntimeError(f"Image analysis failed with both Groq and Gemini: {fallback_exc}") from fallback_exc


def _extract_image(file_path: str) -> str:
    """Extract text and relevant visual information from a standalone image."""
    mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()
    return _extract_vision_from_bytes(
        image_bytes,
        mime_type,
        os.path.basename(file_path),
        max_completion_tokens=800,
    )


def _extract_pdf_page_text(page) -> str:
    """Extract a PDF page in reading order without misclassifying single-column text.

    The previous implementation treated the largest X-coordinate gap as proof of a
    two-column layout. A centered title/target line can create exactly that gap on a
    normal one-column page, which caused later headings to jump ahead of their body.
    """
    blocks = [
        b for b in page.get_text("blocks")
        if len(b) >= 7 and b[4].strip() and b[6] == 0
    ]
    if not blocks:
        plain = page.get_text("text")
        return plain.strip() if plain and plain.strip() else ""

    # PyMuPDF's sorted block order is reliable for ordinary one-column pages.
    sorted_blocks = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))

    # Detect a genuine multi-column layout conservatively. We require at least
    # two substantial blocks on each side of a stable vertical split. A lone
    # centered title/target line is not enough to activate column mode.
    page_width = page.rect.width
    text_blocks = [b for b in blocks if (b[2] - b[0]) < 0.90 * page_width]
    if len(text_blocks) < 4:
        return "\n\n".join(b[4].strip() for b in sorted_blocks)

    xs = sorted(b[0] for b in text_blocks)
    candidate_gaps = []
    for i in range(len(xs) - 1):
        gap = xs[i + 1] - xs[i]
        if gap > 60:
            candidate_gaps.append((gap, (xs[i] + xs[i + 1]) / 2))

    for gap, split_x in sorted(candidate_gaps, reverse=True):
        left = [b for b in text_blocks if b[0] < split_x]
        right = [b for b in text_blocks if b[0] >= split_x]
        if len(left) >= 2 and len(right) >= 2:
            # Avoid treating a single centered line as the start of a column.
            left_width = max((b[2] - b[0]) for b in left)
            right_width = max((b[2] - b[0]) for b in right)
            if left_width > 0.18 * page_width and right_width > 0.18 * page_width:
                left.sort(key=lambda b: (b[1], b[0]))
                right.sort(key=lambda b: (b[1], b[0]))
                full_width = [b for b in blocks if (b[2] - b[0]) >= 0.90 * page_width]
                min_col_y = min(left[0][1], right[0][1])
                top = sorted([b for b in full_width if b[1] < min_col_y], key=lambda b: b[1])
                bottom = sorted([b for b in full_width if b[1] >= min_col_y], key=lambda b: b[1])
                parts = [b[4].strip() for b in top]
                parts.extend(b[4].strip() for b in left)
                parts.extend(b[4].strip() for b in right)
                parts.extend(b[4].strip() for b in bottom)
                return "\n\n".join(p for p in parts if p)

    return "\n\n".join(b[4].strip() for b in sorted_blocks)


def _has_meaningful_page_text(text: str) -> bool:
    """Return True when a page has enough native text to be useful without OCR."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return False
    alnum_count = sum(ch.isalnum() for ch in cleaned)
    # A page with fewer than 50 chars or 15 alnum characters is likely
    # just a header, page number, or watermark on a scanned page.
    if len(cleaned) < 50 or alnum_count < 15:
        return False
    words = [w for w in cleaned.split() if any(c.isalnum() for c in w)]
    if len(words) < 8:
        return False
    return True


def get_or_render_page_image(file_path: str, page_number: int = 1) -> Optional[str]:
    """Return the filesystem path to the preserved page image for visual QA.

    If the document is a standalone image, returns the image path directly.
    If the document is a PDF, checks if the rendered page image already exists in
    the page_images directory. If not, renders and preserves it once, then returns it.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return os.path.abspath(file_path)

    if ext == ".pdf":
        storage_parent = os.path.dirname(os.path.abspath(file_path))
        page_images_dir = os.path.join(storage_parent, "page_images")
        safe_base = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(file_path))
        page_img_path = os.path.join(page_images_dir, f"{safe_base}_page_{page_number}.png")

        if os.path.isfile(page_img_path):
            return page_img_path

        # Render and preserve page image once
        try:
            import pymupdf  # type: ignore
            doc = pymupdf.open(file_path)
            try:
                if 1 <= page_number <= len(doc):
                    page = doc[page_number - 1]
                    _render_pdf_page_for_vision(page, save_path=page_img_path)
                    if os.path.isfile(page_img_path):
                        return page_img_path
            finally:
                doc.close()
        except Exception as exc:
            print(f"[parser] Failed to render page {page_number} for visual QA: {exc}", file=sys.stderr)

    return None


def _render_pdf_page_for_vision(page, save_path: Optional[str] = None) -> tuple[bytes, str]:
    """Render a PDF page to a reasonably sized JPEG/PNG suitable for Vision OCR and preserve on disk."""
    import pymupdf  # type: ignore

    dpi = 220
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            if not os.path.exists(save_path):
                pix.save(save_path)
        except Exception as err:
            print(f"[parser] Non-fatal page image save error: {err}", file=sys.stderr)

    try:
        return pix.tobytes("jpg", jpg_quality=84), "image/jpeg"
    except Exception:
        # PNG is a safe fallback across PyMuPDF versions.
        return pix.tobytes("png"), "image/png"


def _page_has_embedded_images(page) -> bool:
    """Detect non-trivial image objects on a PDF page (screenshots, photos, diagrams)."""
    try:
        images = page.get_images(full=True)
        if not images:
            return False
        for img in images:
            if len(img) >= 4:
                w, h = img[2], img[3]
                # Non-trivial image (not a 16x16 icon, bullet, or thin border)
                if w >= 150 and h >= 150:
                    return True
        return False
    except Exception:
        return False


def _extract_pdf_pymupdf(file_path: str) -> str:
    """Extract native PDF text and Vision-OCR pages that contain scans/images.

    Native text extraction remains the fast path. A page is additionally sent
    through the Vision OCR path when its native text is missing/insufficient
    (scanned PDFs, handwritten pages) or when the page contains substantive images
    with sparse native text. This supports:
    - fully scanned PDFs
    - mixed text + screenshot PDFs
    - PDF pages containing photos/diagrams/charts
    while preserving fast layout-aware extraction for normal PDFs without burning rate limits.
    """
    import pymupdf  # type: ignore

    t_ext_start = time.perf_counter()
    doc = pymupdf.open(file_path)
    total_pages = len(doc)
    print(
        f"[PERF] PDF extraction start - file='{os.path.basename(file_path)}' pages={total_pages}",
        file=sys.stderr,
    )

    page_outputs: list[str] = []
    ocr_failures = 0

    storage_parent = os.path.dirname(os.path.abspath(file_path))
    page_images_dir = os.path.join(storage_parent, "page_images")
    os.makedirs(page_images_dir, exist_ok=True)
    safe_base = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(file_path))

    try:
        for page_number, page in enumerate(doc, start=1):
            native_text = _extract_pdf_page_text(page)
            has_meaningful_text = _has_meaningful_page_text(native_text)
            has_images = _page_has_embedded_images(page)
            # Scanned page / image page needs OCR. A page with substantial text only needs
            # OCR if it has substantive visual content with relatively sparse native text.
            needs_ocr = (not has_meaningful_text) or (has_images and len(native_text.strip()) < 300)

            page_parts: list[str] = []
            if native_text and has_meaningful_text:
                page_parts.append(native_text)

            page_img_path = os.path.join(page_images_dir, f"{safe_base}_page_{page_number}.png")
            ocr_txt_path = os.path.join(page_images_dir, f"{safe_base}_page_{page_number}.ocr.txt")

            if needs_ocr:
                ocr_text = ""
                # Check persistent OCR cache on disk to prevent duplicate OCR
                if os.path.isfile(ocr_txt_path) and os.path.getsize(ocr_txt_path) > 0:
                    try:
                        ocr_text = Path(ocr_txt_path).read_text(encoding="utf-8").strip()
                        print(f"[parser] Loaded cached OCR text for page {page_number}", file=sys.stderr)
                    except Exception as err:
                        print(f"[parser] Failed to read OCR cache for page {page_number}: {err}", file=sys.stderr)

                if not ocr_text:
                    try:
                        t_ocr_start = time.perf_counter()
                        print(f"[PERF] PDF page {page_number} OCR start", file=sys.stderr)
                        image_bytes, mime_type = _render_pdf_page_for_vision(page, save_path=page_img_path)
                        ocr_text = _extract_vision_from_bytes(
                            image_bytes,
                            mime_type,
                            f"{os.path.basename(file_path)} page {page_number}",
                            max_completion_tokens=800,
                        )
                        t_ocr_end = time.perf_counter()
                        print(
                            f"[PERF] PDF page {page_number} OCR end (duration={t_ocr_end - t_ocr_start:.3f}s)",
                            file=sys.stderr,
                        )
                        if ocr_text:
                            try:
                                Path(ocr_txt_path).write_text(ocr_text, encoding="utf-8")
                            except Exception as save_err:
                                print(f"[parser] Non-fatal OCR cache save error: {save_err}", file=sys.stderr)
                        # Add a gentle pause between pages to avoid bursting free-tier rate limits
                        if page_number < total_pages:
                            time.sleep(0.5)
                    except Exception as exc:
                        ocr_failures += 1
                        print(
                            f"[parser] PDF page OCR failed for '{file_path}' page {page_number} (non-fatal): {exc}",
                            file=sys.stderr,
                        )

                if ocr_text:
                    page_parts.append(
                        f"[OCR content from PDF page {page_number}]\n{ocr_text}"
                    )
            elif has_images:
                # Preserve page image on disk for visual questions even if OCR was skipped
                try:
                    _render_pdf_page_for_vision(page, save_path=page_img_path)
                except Exception:
                    pass

            if page_parts:
                page_outputs.append(f"--- Page {page_number} ---\n" + "\n\n".join(page_parts))

        if ocr_failures:
            print(
                f"[parser] PDF Vision OCR completed with {ocr_failures} page failure(s) for '{file_path}'.",
                file=sys.stderr,
            )
        result = "\n\n".join(page_outputs)
        t_ext_end = time.perf_counter()
        print(
            f"[PERF] PDF extraction end (duration={t_ext_end - t_ext_start:.3f}s) - "
            f"pages={len(page_outputs)}/{total_pages}, chars={len(result)}",
            file=sys.stderr,
        )
        return result
    finally:
        doc.close()



def _extract_pdf_pypdf(file_path: str) -> str:
    """Extract text from a PDF using pypdf as fallback."""
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(file_path)
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode="layout")
        except Exception:
            text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _extract_pdf(file_path: str) -> str:
    """Extract PDF text with automatic Vision OCR for scanned/image pages."""
    pymupdf_error = None
    try:
        text = _extract_pdf_pymupdf(file_path)
        if text and text.strip():
            return text
    except Exception as exc:
        pymupdf_error = exc
        print(
            f"[parser] PyMuPDF extraction/OCR failed for '{file_path}': {exc}, falling back to pypdf",
            file=sys.stderr,
        )

    try:
        text = _extract_pdf_pypdf(file_path)
        if text and text.strip():
            return text
    except Exception as exc:
        print(
            f"[parser] pypdf extraction failed for '{file_path}': {exc}",
            file=sys.stderr,
        )

    if pymupdf_error is not None:
        print(
            f"[parser] No native PDF text was recovered from '{file_path}'. "
            "Scanned-page OCR requires PyMuPDF + Groq Vision.",
            file=sys.stderr,
        )
    return ""


def _extract_embedded_images_from_ooxml(
    file_path: str,
    media_prefix: str,
    document_label: str,
) -> list[str]:
    """OCR images embedded inside DOCX/PPTX/XLSX packages."""
    results: list[str] = []
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            members = [
                name
                for name in archive.namelist()
                if name.startswith(media_prefix) and not name.endswith("/")
            ]
            for member in sorted(members):
                data = archive.read(member)
                if len(data) < 4096:
                    # Skip tiny decorative icons, bullets, and borders (< 4KB)
                    continue
                mime_type = mimetypes.guess_type(member)[0] or "image/png"
                try:
                    text = _extract_vision_from_bytes(
                        data,
                        mime_type,
                        f"{document_label} embedded image {os.path.basename(member)}",
                        max_completion_tokens=1200,
                    )
                    if text:
                        results.append(
                            f"[OCR content from embedded image: {os.path.basename(member)}]\n{text}"
                        )
                except Exception as exc:
                    print(
                        f"[parser] Embedded image OCR failed for '{file_path}' image '{member}': {exc}",
                        file=sys.stderr,
                    )
    except zipfile.BadZipFile as exc:
        print(f"[parser] Could not inspect embedded images in '{file_path}': {exc}", file=sys.stderr)

    return results


def _extract_docx(file_path: str) -> str:
    """Extract DOCX text plus OCR from embedded screenshots/images."""
    from docx import Document  # type: ignore

    doc = Document(file_path)
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)

    parts.extend(
        _extract_embedded_images_from_ooxml(
            file_path,
            "word/media/",
            os.path.basename(file_path),
        )
    )
    return "\n\n".join(parts)


def _extract_pptx(file_path: str) -> str:
    """Extract PPTX text plus OCR from embedded screenshots/images."""
    from pptx import Presentation  # type: ignore

    prs = Presentation(file_path)
    parts: list[str] = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        slide_parts: list[str] = [f"--- Slide {slide_num} ---"]
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_parts.append(shape.text.strip())
        if len(slide_parts) > 1:
            parts.append("\n".join(slide_parts))

    parts.extend(
        _extract_embedded_images_from_ooxml(
            file_path,
            "ppt/media/",
            os.path.basename(file_path),
        )
    )
    return "\n\n".join(parts)



def _extract_xlsx(file_path: str) -> str:
    """
    Extract text from an Excel file using pandas + openpyxl.

    Rows are grouped into blocks of ``TABULAR_ROWS_PER_CHUNK`` so that the
    downstream chunker produces far fewer chunks than a row-per-chunk approach,
    keeping total Gemini embedding API calls within free-tier rate limits.
    The column header is prepended to every block so each chunk is self-
    contained and retrieval returns clean, complete context.
    """
    import pandas as pd  # type: ignore

    xl = pd.ExcelFile(file_path, engine="openpyxl")
    parts: list[str] = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        # Build the header line once
        header_line = "  ".join(str(c) for c in df.columns)
        sheet_chunks: list[str] = []
        for start in range(0, len(df), TABULAR_ROWS_PER_CHUNK):
            block = df.iloc[start : start + TABULAR_ROWS_PER_CHUNK]
            rows_text = block.to_string(index=False, header=True, na_rep="")
            sheet_chunks.append(
                f"--- Sheet: {sheet_name} | Rows {start + 1}–"
                f"{min(start + TABULAR_ROWS_PER_CHUNK, len(df))} ---\n"
                + rows_text
            )
        parts.extend(sheet_chunks)

    parts.extend(
        _extract_embedded_images_from_ooxml(
            file_path,
            "xl/media/",
            os.path.basename(file_path),
        )
    )
    return "\n\n".join(parts)


def _extract_csv(file_path: str) -> str:
    """
    Extract text from a CSV file using pandas.

    Rows are grouped into blocks of ``TABULAR_ROWS_PER_CHUNK`` (default 25)
    and emitted as separate paragraphs separated by double newlines.  This
    keeps total chunk count — and therefore total Gemini embedding API calls —
    proportional to ``ceil(row_count / 25)`` rather than ``row_count``.
    The column header is repeated at the top of every block so each chunk is
    fully self-contained for retrieval.
    """
    import pandas as pd  # type: ignore

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return ""

    parts: list[str] = []
    for start in range(0, len(df), TABULAR_ROWS_PER_CHUNK):
        block = df.iloc[start : start + TABULAR_ROWS_PER_CHUNK]
        rows_text = block.to_string(index=False, header=True, na_rep="")
        parts.append(
            f"--- Rows {start + 1}–"
            f"{min(start + TABULAR_ROWS_PER_CHUNK, len(df))} ---\n"
            + rows_text
        )
    return "\n\n".join(parts)


def _extract_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    # Try UTF-8 first, fall back to system default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()
