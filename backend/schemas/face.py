from pydantic import BaseModel

class FaceRequest(BaseModel):
    email: str
    imageData: str
    overwrite: bool = False

class FaceMatchRequest(BaseModel):
    imageData: str

class FaceDeleteRequest(BaseModel):
    email: str
