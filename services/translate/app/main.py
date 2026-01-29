from fastapi import FastAPI
from app.schemas import TranslateRequest, TranslateResponse
from app.model import translate

app = FastAPI(title="Translate Service")

@app.post("/translate", response_model=TranslateResponse)
def translate_text(req: TranslateRequest):
    return translate(req)

@app.get("/health")
def health():
    return {"status": "ok"}
