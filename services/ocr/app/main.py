from fastapi import FastAPI
from app.schemas import OCRRequest, OCRResponse
from app.model import run_ocr

app = FastAPI(title="OCR Service")

@app.post("/ocr", response_model=OCRResponse)
def ocr(req: OCRRequest):
    return run_ocr(req)

@app.get("/health")
def health():
    return {"status": "ok"}
