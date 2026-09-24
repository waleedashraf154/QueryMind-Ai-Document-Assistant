from pydantic import BaseModel

class RecallDocumentRequest(BaseModel):
    chat_id: str
    user_email: str
    filename: str

class DeleteDocumentsRequest(BaseModel):
    user_email: str
    filenames: list[str]
