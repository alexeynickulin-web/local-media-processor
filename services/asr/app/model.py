import whisperx
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {}

def get_model(name: str):
    if name not in MODELS:
        MODELS[name] = whisperx.load_model(
            name,
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8"
        )
    return MODELS[name]

def transcribe(req):
    model = get_model(req.model)
    result = model.transcribe(req.audio_path, language=None)
    segments = result.get("segments") or []
    # Ensure segments match Segment schema (start, end, text only)
    segments_clean = [
        {"start": s["start"], "end": s["end"], "text": s.get("text", "").strip()}
        for s in segments
    ]
    return {
        "language": result.get("language", "en"),
        "text": " ".join(s["text"] for s in segments_clean),
        "segments": segments_clean,
    }
