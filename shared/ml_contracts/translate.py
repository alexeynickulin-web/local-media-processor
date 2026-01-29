from pydantic import BaseModel
from typing import List

class TranslateRequest(BaseModel):
    texts: List[str]
    src_lang: str
    tgt_lang: str

class TranslateResponse(BaseModel):
    translations: List[str]
