from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Response, Request

from backend.schemas.document import RecallDocumentRequest, DeleteDocumentsRequest
from backend.services import documents as service

router = APIRouter(tags=["documents"])


@router.post("/upload/async", summary="Upload a document and start background ingestion")
async def upload_async(
    chat_id: str = Form(...),
    user_email: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Phase 1 only: save the file to disk and Supabase (fire-and-forget),
    then immediately return a file_id. OCR, embedding, and Chroma indexing
    run in a background thread pool. The client polls
    GET /documents/status/{file_id} for progress.
    """
    try:
        return await service.save_document_fast(chat_id, user_email, file)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except service.UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/documents/status/{file_id}", summary="Poll background ingestion status")
def get_document_status(file_id: str):
    """
    Returns the current ingestion status for a file uploaded via /upload/async.

    Status values:
        queued      — waiting for a worker slot in the thread pool
        processing  — OCR / chunking / embedding / indexing in progress
        ready       — document fully indexed; chat queries will now return results
        failed      — ingestion failed; see 'error' field for details
        unknown     — file_id not recognised (server may have restarted)
    """
    return service.get_ingestion_status(file_id)

@router.get("/documents/{user_email}", summary="List all stored documents for a user")
def list_documents(user_email: str):
    try:
        return {"documents": service.list_user_documents(user_email)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}")

@router.post("/documents/delete", summary="Delete stored documents for a user")
def delete_documents(req: DeleteDocumentsRequest):
    try:
        return {"status": "ok", "deleted": service.delete_documents(req.user_email, req.filenames)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {exc}")

@router.get("/documents/{user_email}/file/{filename}", summary="Download/stream a document for preview")
def get_document_file(user_email: str, filename: str, request: Request):
    safe_filename = Path(filename).name
    try:
        file_bytes, _meta, mime_type = service.download_document(user_email, safe_filename)
        total_len = len(file_bytes)
        range_header = request.headers.get("range")
        is_preview = request.query_params.get("preview") == "1" or request.headers.get("x-purpose") == "preview"
        is_download = request.query_params.get("download") == "1"

        if is_preview:
            resp_type = "application/octet-stream" if safe_filename.lower().endswith(".pdf") else mime_type
            resp_disp = "inline"
        elif is_download:
            resp_type = mime_type
            resp_disp = f'attachment; filename="{safe_filename}"'
        else:
            resp_type = mime_type
            resp_disp = f'inline; filename="{safe_filename}"'

        headers = {
            "Content-Disposition": resp_disp,
            "Content-Type": resp_type,
            "Accept-Ranges": "bytes",
        }

        if range_header and range_header.startswith("bytes="):
            range_val = range_header[6:].strip()
            if "," not in range_val:
                parts = range_val.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else total_len - 1
                if start < total_len:
                    end = min(end, total_len - 1)
                    content = file_bytes[start:end + 1]
                    headers["Content-Range"] = f"bytes {start}-{end}/{total_len}"
                    headers["Content-Length"] = str(len(content))
                    return Response(
                        content=content,
                        status_code=206,
                        media_type=mime_type,
                        headers=headers,
                    )

        headers["Content-Length"] = str(total_len)
        return Response(
            content=file_bytes,
            media_type=mime_type,
            headers=headers,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{safe_filename}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {exc}")

@router.get("/documents/{user_email}/preview/{filename}", summary="Get extracted text preview for a document")
def get_document_preview(user_email: str, filename: str):
    safe_filename = Path(filename).name
    try:
        return service.preview_document(user_email, safe_filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{safe_filename}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {exc}")

@router.post("/documents/recall", summary="Recall a stored document into a chat")
def recall_document(req: RecallDocumentRequest):
    try:
        return service.recall_document(req.chat_id, req.user_email, req.filename)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError:
        safe_filename = Path(req.filename).name
        raise HTTPException(status_code=404, detail=f"Document '{safe_filename}' not found in stored documents.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve '{Path(req.filename).name}' from persistent storage: {exc}")

@router.post("/upload", summary="Upload and index a document")
async def upload(chat_id: str = Form(...), user_email: str = Form(...), file: UploadFile = File(...)):
    try:
        return await service.upload_document(chat_id, user_email, file)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except service.UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

