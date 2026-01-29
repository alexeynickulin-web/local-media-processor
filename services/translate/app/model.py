import torch
from transformers import pipeline

PIPELINES = {}
DEVICE = 0 if torch.cuda.is_available() else -1


def get_pipeline():
    if "nllb" not in PIPELINES:
        PIPELINES["nllb"] = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            device=DEVICE,
        )
    return PIPELINES["nllb"]

def translate(req):
    pipe = get_pipeline()
    result = pipe(
        req.texts,
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang
    )
    return {
        "translations": [r["translation_text"] for r in result]
    }
