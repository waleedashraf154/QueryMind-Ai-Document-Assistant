from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    chat_id: str
    user_email: str
    question: str
    selected_source: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    chat_title: Optional[str] = None

class CreateChatRequest(BaseModel):
    user_email: str
    chat_id: str
    title: str = "New Chat"

class DeleteChatRequest(BaseModel):
    chat_id: str
    user_email: str

class RenameChatRequest(BaseModel):
    chat_id: str
    user_email: str
    title: str
