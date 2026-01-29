import os

ASR_URL = os.getenv("ASR_URL", "http://asr:8000/transcribe")
TRANSLATE_URL = os.getenv("TRANSLATE_URL", "http://translate:8000/translate")
OCR_URL = os.getenv("OCR_URL", "http://ocr:8000/ocr")
TTS_URL = os.getenv("TTS_URL", "http://tts:8000/tts")
DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR", "/data/output")
