from pydantic import BaseModel


class TTSRequest(BaseModel):
    text: str
    voice: str
    output_path: str


class TTSResponse(BaseModel):
    output_path: str
