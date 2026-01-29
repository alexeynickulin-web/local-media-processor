from pydantic import BaseModel
from typing import List


class OCRRequest(BaseModel):
    image_path: str
    languages: List[str]


class OCRResponse(BaseModel):
    text: str
