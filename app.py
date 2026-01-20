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
    if not text:
        return ""
    if target_lang == "en":
        return translator(text)[0]['translation_text']
    else:
        en_text = translator(text)[0]['translation_text']
        # Для перевода из EN в target используйте другую модель, если нужно; здесь пример с ru
        target_translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_lang}")
        return target_translator(en_text)[0]['translation_text']

def text_to_speech(text, target_lang="en"):
    """TTS."""
    if not text:
        return None
    output_path = tempfile.mktemp(suffix=".wav")
    tts_model.tts_to_file(text=text, file_path=output_path, speaker="default", language=target_lang)
    return output_path

def process_media(input_type, file, input_text, media_type, source_lang, target_lang, do_transcribe, do_translate, do_tts):
    """Основная функция обработки с выбором действий."""
    text = ""
    translated_text = ""
    tts_audio = None
    detected_lang = source_lang
    
    if do_transcribe:
        if input_type != "File" or not file:
            return "Для транскрипции нужен файл (аудио/видео/изображение).", None, None
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
            return "Неверный тип медиа.", None, None
    else:
        if input_type == "Text" and input_text:
            text = input_text
        else:
            return "Для перевода/TTS нужен текст (или транскрипция).", None, None
    
    if do_translate:
        translated_text = translate_text(text, target_lang)
    else:
        translated_text = text  # Если перевод не нужен, используем оригинал для TTS
    
    if do_tts:
        tts_audio = text_to_speech(translated_text, target_lang)
    
    result_text = f"Оригинальный текст ({detected_lang}): {text}\n"
    if do_translate:
        result_text += f"Переведённый ({target_lang}): {translated_text}\n"
    
    return result_text, tts_audio, tts_audio  # Для скачивания

# Gradio интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# Local Media Processor (Flexible Actions)")
    
    input_type = gr.Radio(choices=["File", "Text"], label="Тип входа", value="File")
    file_input = gr.File(label="Загрузите аудио/видео/изображение (для File)")
    input_text = gr.Textbox(label="Входной текст (для Text)", visible=False)
    media_type = gr.Dropdown(choices=["Audio", "Video", "Image"], label="Тип медиа (для File)", visible=True)
    
    source_lang = gr.Textbox(label="Исходный язык (e.g., en, ru, auto для аудио)")
    target_lang = gr.Textbox(label="Целевой язык (e.g., en, ru)")
    
    do_transcribe = gr.Checkbox(label="Выполнить транскрипцию/OCR", value=True)
    do_translate = gr.Checkbox(label="Выполнить перевод", value=True)
    do_tts = gr.Checkbox(label="Выполнить TTS (текст в речь)", value=True)
    
    process_btn = gr.Button("Обработать")
    
    output_text = gr.Textbox(label="Результат текста")
    output_audio = gr.Audio(label="TTS Выход")
    output_download = gr.File(label="Скачать TTS аудио")
    
    # Динамическая видимость
    def update_visibility(input_type):
        file_visible = input_type == "File"
        text_visible = input_type == "Text"
        media_visible = file_visible
        return (
            gr.update(visible=file_visible),
            gr.update(visible=text_visible),
            gr.update(visible=media_visible)
        )
    
    input_type.change(update_visibility, inputs=[input_type], outputs=[file_input, input_text, media_type])
    
    process_btn.click(
        process_media, 
        inputs=[input_type, file_input, input_text, media_type, source_lang, target_lang, do_transcribe, do_translate, do_tts], 
        outputs=[output_text, output_audio, output_download]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
