import httpx
import os

from app.config import ASR_URL, TRANSLATE_URL, OCR_URL, TTS_URL, DATA_OUTPUT_DIR
from app.schemas import ProcessResponse


async def run_pipeline(file_path: str, file_type: str) -> ProcessResponse:
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    async with httpx.AsyncClient(timeout=300) as client:
        # OCR for images
        if file_type == "image":
            ocr_resp = await client.post(
                OCR_URL,
                json={"image_path": file_path, "languages": ["en", "ru"]},
            )
            ocr_resp.raise_for_status()
            data = ocr_resp.json()
            text = data["text"]
        else:
            # ASR for audio/video
            asr_resp = await client.post(
                ASR_URL,
                json={
                    "audio_path": file_path,
                    "language": "auto",
                    "model": "medium",
                },
            )
            asr_resp.raise_for_status()
            data = asr_resp.json()
            text = data["text"]

        # Translate
        tr_resp = await client.post(
            TRANSLATE_URL,
            json={
                "texts": [text],
                "src_lang": "eng_Latn",
                "tgt_lang": "rus_Cyrl",
            },
        )
        tr_resp.raise_for_status()
        translated = tr_resp.json()["translations"][0]

        # TTS
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_mp3 = os.path.join(DATA_OUTPUT_DIR, f"{base_name}.mp3")
        tts_resp = await client.post(
            TTS_URL,
            json={
                "text": translated,
                "voice": "ru-RU-SvetlanaNeural",
                "output_path": out_mp3,
            },
        )
        tts_resp.raise_for_status()

    return ProcessResponse(
        original_text=text,
        translated_text=translated,
        audio_path=out_mp3,
    )
