import edge_tts
import os

async def synthesize(req):
    os.makedirs(os.path.dirname(req.output_path), exist_ok=True)

    communicate = edge_tts.Communicate(
        text=req.text,
        voice=req.voice
    )
    await communicate.save(req.output_path)

    return {
        "output_path": req.output_path
    }
