from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.services.transcription import transcribe_audio

router = APIRouter(tags=["transcription"])

@router.post("/transcribe", summary="Transcribe speech audio into English text")
async def transcribe(file: UploadFile = File(...)):
    try:
        return await transcribe_audio(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
