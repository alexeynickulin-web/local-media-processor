from fastapi import FastAPI
from app.schemas import TTSRequest, TTSResponse
from app.model import synthesize

app = FastAPI(title="TTS Service")

@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    return await synthesize(req)

@app.get("/health")
def health():
    return {"status": "ok"}
