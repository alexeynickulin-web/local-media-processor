from fastapi import FastAPI
from ml_contracts.asr import ASRRequest, ASRResponse
from app.model import transcribe

app = FastAPI(title="ASR Service")

@app.post("/transcribe", response_model=ASRResponse)
def transcribe_audio(req: ASRRequest):
    return transcribe(req)

@app.get("/health")
def health():
    return {"status": "ok"}
