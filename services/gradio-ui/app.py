"""Gradio UI service - Main application."""

import os
import time
import logging

import gradio as gr
import httpx

logger = logging.getLogger("gradio-ui")

API_GATEWAY_URL = os.environ.get("API_GATEWAY_URL", "http://api-gateway:8000")
TTS_SERVICE_URL = os.environ.get("TTS_SERVICE_URL", "http://tts:8000")
API_BASE = f"{API_GATEWAY_URL}/api"
POLL_INTERVAL = 2
MAX_RETRIES = 30


class APIClient:
    """Client for API Gateway."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api"
        self.client = httpx.AsyncClient(timeout=300.0)

    async def check_services_health(self) -> dict:
        """Check which services are available."""
        try:
            response = await self.client.get(f"{self.api_base}/services/health")
            response.raise_for_status()
            data = response.json()
            return data.get("services", {})
        except Exception as e:
            logger.warning(f"Could not check services: {e}")
            return {}

    async def submit_transcribe(self, file_path: str, model: str, language: str) -> dict:
        """Submit transcription task."""
        response = await self.client.post(
            f"{self.api_base}/transcribe",
            json={
                "file_path": file_path,
                "model_name": model,
                "language": language,
            },
        )
        response.raise_for_status()
        return response.json()

    async def submit_translate(self, text: str, src_lang: str, tgt_lang: str, model: str) -> dict:
        """Submit translation task."""
        response = await self.client.post(
            f"{self.api_base}/translate",
            json={
                "text": text,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "model_name": model,
            },
        )
        response.raise_for_status()
        return response.json()

    async def submit_ocr(self, file_path: str, language: str) -> dict:
        """Submit OCR task."""
        response = await self.client.post(
            f"{self.api_base}/ocr",
            json={
                "file_path": file_path,
                "language": language,
            },
        )
        response.raise_for_status()
        return response.json()

    async def submit_tts(self, text: str, language: str, voice: str | None) -> dict:
        """Submit TTS task."""
        response = await self.client.post(
            f"{self.api_base}/tts",
            json={
                "text": text,
                "language": language,
                "voice": voice,
            },
        )
        response.raise_for_status()
        return response.json()

    async def submit_subtitle(self, segments: list, format: str = "srt") -> dict:
        """Submit subtitle generation task."""
        response = await self.client.post(
            f"{self.api_base}/subtitle",
            json={
                "segments": segments,
                "format": format,
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_task_status(self, task_id: str) -> dict:
        """Get task status."""
        response = await self.client.get(f"{self.api_base}/status/{task_id}")
        response.raise_for_status()
        return response.json()

    async def submit_tts_direct(self, text: str, language: str, voice: str | None) -> dict:
        """Submit TTS task directly to TTS service."""
        response = await self.client.post(
            f"{TTS_SERVICE_URL}/tts",
            json={
                "text": text,
                "language": language,
                "voice": voice,
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_tts_status_direct(self, task_id: str) -> dict:
        """Get TTS task status directly from TTS service."""
        response = await self.client.get(f"{TTS_SERVICE_URL}/status/{task_id}")
        response.raise_for_status()
        return response.json()


api_client = APIClient(API_GATEWAY_URL)


async def poll_task_status(task_id: str) -> dict:
    """Poll task until completion."""
    for _ in range(MAX_RETRIES):
        status_data = await api_client.get_task_status(task_id)
        status = status_data.get("status", "")

        if status == "completed":
            return status_data
        elif status == "failed":
            return status_data

        time.sleep(POLL_INTERVAL)

    return {"status": "timeout", "error": "Task timed out"}


async def check_services_status() -> str:
    """Check which services are running."""
    services = await api_client.check_services_health()
    
    if not services:
        return "⚠️ Could not connect to API Gateway. Is it running?"
    
    lines = ["**Available Services:**"]
    for name, status in services.items():
        icon = "✅" if status else "❌"
        lines.append(f"{icon} {name}")
    
    return "\n".join(lines)


async def process_transcribe(
    file_path: str,
    model: str,
    language: str,
    do_translate: bool,
    tgt_lang: str,
    do_tts: bool,
) -> tuple[str, str]:
    """Process transcription with optional translation and TTS."""
    try:
        result = await api_client.submit_transcribe(file_path, model, language)
        task_id = result.get("task_id")

        status_data = await poll_task_status(task_id)
        transcribe_result = status_data.get("result", {})

        text = transcribe_result.get("text", "")
        if not text:
            return "❌ Transcription failed", ""

        result_text = f"📝 Transcription:\n{text}\n\n"

        if do_translate and text:
            trans_result = await api_client.submit_translate(
                text, language, tgt_lang, "facebook/nllb-200-distilled-600M"
            )
            trans_task_id = trans_result.get("task_id")

            trans_status = await poll_task_status(trans_task_id)
            trans_data = trans_status.get("result", {})

            translated = trans_data.get("translated_text", "")
            if translated:
                result_text += f"🌐 Translation ({tgt_lang}):\n{translated}\n\n"

                if do_tts:
                    tts_result = await api_client.submit_tts(translated, tgt_lang, None)
                    tts_task_id = tts_result.get("task_id")

                    tts_status = await poll_task_status(tts_task_id)
                    tts_data = tts_status.get("result", {})

                    audio_path = tts_data.get("audio_path", "")
                    if audio_path:
                        result_text += f"🔊 TTS generated: {audio_path}"

        return result_text, f"✅ Task ID: {task_id}"

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return f"❌ Error: {e}", ""


async def process_only_translate(text: str, src_lang: str, tgt_lang: str) -> tuple[str, str]:
    """Process translation only."""
    try:
        result = await api_client.submit_translate(text, src_lang, tgt_lang, "facebook/nllb-200-distilled-600M")
        task_id = result.get("task_id")

        status_data = await poll_task_status(task_id)
        trans_data = status_data.get("result", {})

        translated = trans_data.get("translated_text", "")
        if translated:
            return translated, f"✅ Task ID: {task_id}"

        return "❌ Translation failed", ""

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"❌ Error: {e}", ""


async def process_only_ocr(file_path: str, language: str) -> tuple[str, str]:
    """Process OCR only."""
    try:
        result = await api_client.submit_ocr(file_path, language)
        task_id = result.get("task_id")

        status_data = await poll_task_status(task_id)
        ocr_data = status_data.get("result", {})

        text = ocr_data.get("text", "")
        if text:
            return text, f"✅ Task ID: {task_id}"

        return "❌ OCR failed", ""

    except Exception as e:
        logger.error(f"OCR error: {e}")
        return f"❌ Error: {e}", ""


async def poll_tts_status(task_id: str) -> dict:
    """Poll TTS status until completion."""
    for _ in range(MAX_RETRIES):
        try:
            status_data = await api_client.get_tts_status_direct(task_id)
            status = status_data.get("status", "")

            if status == "completed":
                return status_data
            elif status == "failed":
                return status_data

        except Exception:
            pass

        time.sleep(POLL_INTERVAL)

    return {"status": "timeout", "error": "Task timed out"}


async def process_only_tts(text: str, language: str, voice: str | None) -> tuple[str, str]:
    """Process TTS only."""
    try:
        result = await api_client.submit_tts_direct(text, language, voice)
        task_id = result.get("task_id")

        status_data = await poll_tts_status(task_id)

        if status_data.get("status") == "completed":
            tts_data = status_data.get("result", {})
            audio_path = tts_data.get("audio_path", "")
            if audio_path:
                return audio_path, f"✅ Task ID: {task_id}"

        return "❌ TTS failed", ""

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return f"❌ Error: {e}", ""


async def get_tts_audio(task_id: str) -> str | None:
    """Get audio file path from TTS result."""
    try:
        status_data = await api_client.get_tts_status_direct(task_id)
        if status_data.get("status") == "completed":
            tts_data = status_data.get("result", {})
            audio_path = tts_data.get("audio_path", "")
            return audio_path
    except Exception as e:
        logger.error(f"Error getting audio: {e}")
    return None


async def process_only_tts_audio(text: str, language: str, voice: str) -> tuple[str, str]:
    """Process TTS and return download URL."""
    try:
        voice_value = None if voice == "auto" else voice
        result = await api_client.submit_tts_direct(text, language, voice_value)
        task_id = result.get("task_id")

        status_data = await poll_tts_status(task_id)

        if status_data.get("status") == "completed":
            return f"http://localhost:8004/download/{task_id}", f"✅ Task ID: {task_id}"

        return "❌ TTS failed", ""

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return f"❌ Error: {e}", ""


def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(title="Local Media Processor") as interface:
        gr.Markdown("# 🎬 Local Media Processor")
        gr.Markdown(f"**API Gateway:** `{API_GATEWAY_URL}`")
        gr.Markdown("---")
        
        gr.Markdown("**Quick Start:** Click 'Check Services' to see what's running, then use the tabs below.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔄 Services Status")
                check_btn = gr.Button("Check Services", variant="secondary")
            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Status", lines=4, interactive=False)

        check_btn.click(
            fn=check_services_status,
            inputs=[],
            outputs=[status_output],
        )

        gr.Markdown("---")

        with gr.Tab("Transcribe + Translate + TTS"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📁 Input")
                    file_input = gr.File(label="Upload Audio/Video", file_count="single")
                    src_lang = gr.Dropdown(
                        ["auto", "en", "ru", "fr", "de", "es"],
                        label="Source Language", value="auto",
                    )

                    gr.Markdown("### ⚙️ Options")
                    w_model = gr.Dropdown(
                        ["tiny", "base", "small", "medium", "large-v3"],
                        label="Whisper Model", value="medium",
                    )
                    do_translate = gr.Checkbox(label="Translate", value=True)
                    tgt_lang = gr.Dropdown(
                        ["ru", "en", "de", "es", "fr"], label="Target Language", value="ru"
                    )
                    do_tts = gr.Checkbox(label="Generate TTS", value=True)

                    transcribe_btn = gr.Button("🚀 Process", variant="primary")

                with gr.Column():
                    gr.Markdown("### 📝 Results")
                    result_output = gr.Textbox(label="Output", lines=12)
                    status_output = gr.Textbox(label="Status", lines=2)

            transcribe_btn.click(
                fn=process_transcribe,
                inputs=[file_input, w_model, src_lang, do_translate, tgt_lang, do_tts],
                outputs=[result_output, status_output],
            )

        with gr.Tab("Translate Only"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📝 Input Text")
                    text_input = gr.Textbox(label="Text to Translate", lines=5)
                    src_lang_t = gr.Dropdown(
                        ["en", "ru", "fr", "de", "es", "zh", "ja"],
                        label="Source Language", value="en",
                    )
                    tgt_lang_t = gr.Dropdown(
                        ["ru", "en", "de", "es", "fr", "zh", "ja"],
                        label="Target Language", value="ru",
                    )
                    translate_btn = gr.Button("🌐 Translate", variant="primary")

                with gr.Column():
                    gr.Markdown("### 🌐 Translation")
                    trans_output = gr.Textbox(label="Translated", lines=10)
                    status_t = gr.Textbox(label="Status", lines=2)

            translate_btn.click(
                fn=process_only_translate,
                inputs=[text_input, src_lang_t, tgt_lang_t],
                outputs=[trans_output, status_t],
            )

        with gr.Tab("OCR Only"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📷 Image")
                    ocr_file = gr.File(label="Upload Image", file_count="single")
                    ocr_lang = gr.Dropdown(
                        ["en", "ru", "en_ru", "fr", "de", "es"],
                        label="Language", value="en",
                    )
                    ocr_btn = gr.Button("🔍 OCR", variant="primary")

                with gr.Column():
                    gr.Markdown("### 📄 Result")
                    ocr_output = gr.Textbox(label="Extracted Text", lines=10)
                    ocr_status = gr.Textbox(label="Status", lines=2)

            ocr_btn.click(
                fn=process_only_ocr,
                inputs=[ocr_file, ocr_lang],
                outputs=[ocr_output, ocr_status],
            )

        with gr.Tab("TTS Only"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔊 Text")
                    tts_text = gr.Textbox(label="Text", lines=5)
                    tts_lang = gr.Dropdown(
                        ["en", "ru", "fr", "de", "es", "zh", "ja", "it"],
                        label="Language", value="en",
                    )
                    tts_voice = gr.Dropdown(
                        [
                            "auto",
                            "en-US-JennyNeural",
                            "en-US-GuyNeural",
                            "en-US-AriaNeural",
                            "en-GB-SoniaNeural",
                            "en-GB-RyanNeural",
                            "ru-RU-SvetlanaNeural",
                            "ru-RU-DmitryNeural",
                            "fr-FR-DeniseNeural",
                            "fr-FR-HenriNeural",
                            "de-DE-KlaudiaNeural",
                            "de-DE-ConradNeural",
                            "es-ES-ElviraNeural",
                            "es-MX-DaliaNeural",
                            "zh-CN-XiaoxiaoNeural",
                            "zh-CN-YunxiNeural",
                            "ja-JP-NanamiNeural",
                            "ja-JP-KeiichiNeural",
                            "it-IT-ElsaNeural",
                            "it-IT-DiegoNeural",
                        ],
                        label="Voice",
                        value="auto",
                    )
                    tts_btn = gr.Button("🔊 Generate", variant="primary")

                with gr.Column():
                    gr.Markdown("### 🎵 Audio")
                    tts_output = gr.Textbox(label="Download Link", lines=3)
                    tts_status = gr.Textbox(label="Status", lines=2)

            tts_btn.click(
                fn=process_only_tts_audio,
                inputs=[tts_text, tts_lang, tts_voice],
                outputs=[tts_output, tts_status],
            )

        gr.Markdown("---")
        gr.Markdown(f"*Gateway: {API_GATEWAY_URL}* | *Ports: Gateway=8000, Gradio=7860*")

    return interface


app = build_interface()

if __name__ == "__main__":
    logger.info(f"Starting Gradio UI, Gateway: {API_GATEWAY_URL}")
    app.launch(server_name="0.0.0.0", server_port=7860)