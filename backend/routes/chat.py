from fastapi import APIRouter, HTTPException
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.chat import answer_chat

router = APIRouter(tags=["chat"])

@router.post("/chat", response_model=ChatResponse, summary="Ask a question about uploaded documents")
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return answer_chat(req.chat_id, req.user_email, req.question, req.selected_source)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {exc}")
