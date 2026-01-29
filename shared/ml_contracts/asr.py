from pydantic import BaseModel
from typing import List
from .segment import Segment

class ASRRequest(BaseModel):
    audio_path: str
    language: str = "auto"
    model: str = "medium"

class ASRResponse(BaseModel):
    language: str
    text: str
    segments: List[Segment]
