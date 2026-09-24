from fastapi import APIRouter, HTTPException
from backend.schemas.face import FaceRequest, FaceMatchRequest, FaceDeleteRequest
from backend.services import face as service

router = APIRouter(tags=["face"])

@router.post("/face/register", summary="Register one Face ID")
async def face_register(req: FaceRequest):
    try:
        result = await service.register(req.email, req.imageData, overwrite=req.overwrite)
        return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Face registration timed out. Please try again.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face registration failed: {exc}")

@router.post("/face/match", summary="Match a face to a registered account")
def face_match(req: FaceMatchRequest):
    try:
        return service.match(req.imageData)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face matching failed: {exc}")

@router.get("/face/status", summary="Check if a user has a Face ID registered")
def face_status(email: str):
    try:
        return service.status(email)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not check Face ID status: {exc}")

@router.delete("/face/delete", summary="Delete a user's registered Face ID")
def face_delete(req: FaceDeleteRequest):
    try:
        return service.delete(req.email)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face deletion failed: {exc}")
