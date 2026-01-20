import gradio as gr
from faster_whisper import WhisperModel
import easyocr
from transformers import pipeline
from TTS.api import TTS
import torch
import os
import moviepy.editor as mp
import tempfile

# Определение устройства (GPU если доступно)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Загрузка моделей (скачиваются автоматически при первом запуске)
whisper_model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
ocr_reader = easyocr.Reader(['en', 'ru', 'fr', 'de', 'es', 'zh', 'ja'])  # Добавьте нужные языки
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")  # Многоязычный перевод в/из EN; для других пар замените модель
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

def extract_audio_from_video(video_path):
    """Извлечение аудио из видео."""
    video = mp.VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path, source_lang="auto"):
    """Транскрипция аудио."""
    segments, info = whisper_model.transcribe(audio_path, language=source_lang if source_lang != "auto" else None)
    text = " ".join([segment.text for segment in segments])
    detected_lang = info.language
    return text, detected_lang

def ocr_image(image_path, source_lang="en"):
    """OCR для изображения."""
    result = ocr_reader.readtext(image_path, detail=0, lang_list=[source_lang])
    text = " ".join(result)
    return text

def translate_text(text, target_lang="en"):
    """Перевод текста (через EN как pivot, если нужно)."""
    # Для простоты: переводим в EN, затем в target, если target != en
    if target_lang == "en":
        return translator(text)[0]['translation_text']
    else:
        en_text = translator(text)[0]['translation_text']
        # Для перевода из EN в target используйте другую модель, если нужно; здесь пример с ru
        target_translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_lang}")
        return target_translator(en_text)[0]['translation_text']

def text_to_speech(text, target_lang="en"):
    """TTS."""
    output_path = tempfile.mktemp(suffix=".wav")
    tts_model.tts_to_file(text=text, file_path=output_path, speaker="default", language=target_lang)
    return output_path

def process_media(file, media_type, source_lang, target_lang):
    """Основная функция обработки."""
    if media_type == "Audio":
        text, detected_lang = transcribe_audio(file, source_lang)
    elif media_type == "Video":
        audio_path = extract_audio_from_video(file)
        text, detected_lang = transcribe_audio(audio_path, source_lang)
        os.remove(audio_path)
    elif media_type == "Image":
        text = ocr_image(file, source_lang)
        detected_lang = source_lang  # Для OCR нет авто-детекции
    else:
        return "Invalid media type", None, None
    
    translated_text = translate_text(text, target_lang)
    tts_audio = text_to_speech(translated_text, target_lang)
    
    return f"Original text ({detected_lang}): {text}\nTranslated ({target_lang}): {translated_text}", tts_audio, translated_text

# Gradio интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# Local Media Processor")
    file_input = gr.File(label="Upload Audio/Video/Image")
    media_type = gr.Dropdown(choices=["Audio", "Video", "Image"], label="Media Type")
    source_lang = gr.Textbox(label="Source Language (e.g., en, ru, auto for audio)")
    target_lang = gr.Textbox(label="Target Language (e.g., en, ru)")
    process_btn = gr.Button("Process")
    
    output_text = gr.Textbox(label="Result Text")
    output_audio = gr.Audio(label="TTS Output")
    output_download = gr.File(label="Download TTS Audio")
    
    process_btn.click(process_media, inputs=[file_input, media_type, source_lang, target_lang], outputs=[output_text, output_audio, output_download])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
