from pydantic import BaseModel
from typing import Literal, Optional


class ProcessRequest(BaseModel):
    file_path: str
    file_type: Optional[Literal["audio", "image"]] = None  # inferred from path if omitted


class ProcessResponse(BaseModel):
    original_text: str
    translated_text: str
    audio_path: str
