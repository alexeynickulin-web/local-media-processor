import easyocr
import torch

READERS = {}
DEVICE = torch.cuda.is_available()

def get_reader(langs):
    key = tuple(sorted(langs))
    if key not in READERS:
        READERS[key] = easyocr.Reader(
            list(langs),
            gpu=DEVICE,
            model_storage_directory="/models/ocr"
        )
    return READERS[key]

def run_ocr(req):
    reader = get_reader(req.languages)
    result = reader.readtext(req.image_path, detail=0, paragraph=True)
    return {
        "text": " ".join(result)
    }
