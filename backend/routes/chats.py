from fastapi import APIRouter, HTTPException
from backend.schemas.chat import CreateChatRequest, DeleteChatRequest, RenameChatRequest
from backend.services import chats as service

router = APIRouter(tags=["chats"])

@router.post("/chats/create", summary="Create a new chat record")
def create_chat(req: CreateChatRequest):
    try:
        service.create_chat(req.user_email, req.chat_id, req.title)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {exc}")
    return {"status": "ok", "chat_id": req.chat_id}

@router.get("/chats/{user_email}", summary="Load all chats for a user")
def get_chats(user_email: str):
    try:
        return service.get_chats(user_email)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load chats: {exc}")

@router.get("/chats/{chat_id}/sources", summary="Get documents and sources for a specific chat")
def get_chat_sources(chat_id: str, user_email: str | None = None):
    try:
        return service.get_sources(chat_id, user_email)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

@router.delete("/chats/{chat_id}", summary="Delete a chat")
def delete_chat(chat_id: str):
    raise HTTPException(status_code=405, detail="Use POST /chats/delete with the authenticated user.")

@router.post("/chats/delete", summary="Delete a chat via POST")
def delete_chat_post(req: DeleteChatRequest):
    try:
        return service.delete_chat(req.chat_id, req.user_email)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/chats/rename", summary="Rename a chat via POST")
def rename_chat_post(req: RenameChatRequest):
    try:
        return service.rename_chat(req.chat_id, req.user_email, req.title)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to rename chat: {exc}")
