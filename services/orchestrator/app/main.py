from fastapi import FastAPI, HTTPException

from app.pipeline import run_pipeline
from app.schemas import ProcessRequest, ProcessResponse

app = FastAPI(title="Orchestrator")


def _infer_file_type(file_path: str) -> str:
    ext = (file_path or "").lower().split(".")[-1]
    if ext in ("jpg", "jpeg", "png", "bmp", "gif", "webp"):
        return "image"
    return "audio"


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    file_type = req.file_type or _infer_file_type(req.file_path)
    try:
        return await run_pipeline(req.file_path, file_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
