import gradio as gr
from faster_whisper import WhisperModel
import easyocr
from transformers import pipeline
from TTS.api import TTS
import torch
import os
import moviepy.editor as mp
import tempfile
from functools import lru_cache
import time
from datetime import datetime
import fasttext  # ← новый импорт

# ==================== НАСТРОЙКА ПУТЕЙ К МОДЕЛЯМ ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for subdir in ["whisper", "tts", "translation", "ocr", "huggingface", "fasttext"]:
    os.makedirs(os.path.join(MODELS_DIR, subdir), exist_ok=True)

os.environ["HF_HUB_CACHE"] = os.path.join(MODELS_DIR, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODELS_DIR, "huggingface")
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "huggingface")
os.environ["COQUI_TTS_CACHE"] = os.path.join(MODELS_DIR, "tts")

FASTTEXT_MODEL_PATH = os.path.join(MODELS_DIR, "fasttext", "lid.176.bin")

print(f"Модели сохраняются в: {MODELS_DIR}")

# Устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Загрузка fastText модели (один раз при старте)
fasttext_model = None
if not os.path.exists(FASTTEXT_MODEL_PATH):
    print("Модель fastText lid.176.bin не найдена. Скачиваем автоматически...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)
    print("fastText модель загружена.")

fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
print("fastText модель загружена для language detection.")

# Доступные модели (остальное без изменений)
WHISPER_MODELS = {
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "medium": "medium",
    "small": "small",
    "distil-large-v3": "Systran/faster-whisper-distil-large-v3",
}

TTS_MODELS = {
    "xtts_v2 (multilingual)": "tts_models/multilingual/multi-dataset/xtts_v2",
    "your_tts (multilingual)": "tts_models/multilingual/multi-dataset/your_tts",
    "en/tacotron2-DDC": "tts_models/en/ljspeech/tacotron2-DDC",
    "ru/vits": "tts_models/ru/multi-dataset/vits",
}

TRANSLATION_MODELS_PIVOT = [
    "Helsinki-NLP/opus-mt-mul-en",
    "Helsinki-NLP/opus-mt-tc-big-mul-en",
]

# Глобальные переменные
whisper_model = None
tts_model = None
current_whisper_name = None
current_tts_name = None
model_status_text = "Модели не загружены"
ocr_reader = easyocr.Reader(
    ['en', 'ru', 'fr', 'de', 'es', 'zh', 'ja'],
    download_enabled=True,
    model_storage_directory=os.path.join(MODELS_DIR, "ocr"),
    user_network_directory=os.path.join(MODELS_DIR, "ocr")
)

@lru_cache(maxsize=32)
def get_translator(model_name):
    return pipeline(
        "translation",
        model=model_name,
        cache_dir=os.path.join(MODELS_DIR, "translation"),
        device=device if device == "cuda" else -1
    )

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def timed_step(step_name, func, *args, **kwargs):
    start = time.time()
    log(f"Начало: {step_name}")
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    log(f"Завершено: {step_name} → {elapsed:.2f} сек")
    return result, elapsed

# Новая функция: определение языка с fastText
def detect_language_fasttext(text):
    if not text.strip():
        return "unknown", 0.0
    # fastText ожидает список строк
    prediction = fasttext_model.predict([text], k=1)
    lang_label = prediction[0][0][0]          # '__label__ru'
    prob = prediction[1][0][0]                # вероятность
    lang_code = lang_label.replace('__label__', '')
    return lang_code, prob

# Остальные функции load_whisper, load_tts, load_selected_models — без изменений
# (пропускаю их для краткости, копируйте из предыдущей версии)

# ────────────────────────────────────────────────
# Функции обработки (обновлён OCR блок)
# ────────────────────────────────────────────────

def extract_audio_from_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path, source_lang="auto"):
    if whisper_model is None:
        raise ValueError("Модель Whisper не загружена")
    lang_param = None if source_lang.lower() == "auto" else source_lang
    segments, info = whisper_model.transcribe(
        audio_path,
        language=lang_param,
        beam_size=5,
        vad_filter=True
    )
    text = " ".join([segment.text for segment in segments])
    return text, info.language, info.language_probability

def ocr_image(image_path, source_lang="en"):
    result = ocr_reader.readtext(image_path, detail=0, paragraph=True, lang_list=[source_lang])
    return " ".join(result)

def translate_text(text, source_lang, target_lang, pivot_model):
    if not text.strip():
        return ""
    translator_pivot = get_translator(pivot_model)
    en_text = translator_pivot(text, src_lang=source_lang)[0]['translation_text'] if source_lang.lower() != "en" else text
    if target_lang.lower() == "en":
        return en_text
    target_model = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    try:
        translator_target = get_translator(target_model)
        return translator_target(en_text)[0]['translation_text']
    except Exception as e:
        return f"[Ошибка перевода в {target_lang}: {str(e)}]"

def text_to_speech(text, target_lang="en"):
    if tts_model is None or not text.strip():
        return None
    output_path = tempfile.mktemp(suffix=".wav")
    try:
        tts_model.tts_to_file(text=text, file_path=output_path, speaker="default", language=target_lang)
    except:
        tts_model.tts_to_file(text=text, file_path=output_path, speaker="default")
    return output_path

def process_media(input_type, file, input_text, media_type, source_lang, target_lang,
                  do_transcribe, do_translate, do_tts, whisper_model_key, tts_model_key, pivot_model):
    
    start_total = time.time()
    timings = []
    warnings = []
    
    if do_transcribe and whisper_model is None:
        warnings.append("Модель Whisper не загружена!")
    if do_tts and tts_model is None:
        warnings.append("Модель TTS не загружена!")
    
    if warnings:
        return "\n".join(warnings) + "\n\n" + model_status_text, None, None
    
    text = ""
    translated_text = ""
    tts_audio = None
    detected_lang = source_lang
    detected_prob = 0.0
    audio_path = None
    
    log("─" * 60)
    log(f"Начало обработки | {input_type} | {media_type or ''} | actions: {do_transcribe=}, {do_translate=}, {do_tts=}")
    
    # 1. Извлечение аудио
    if do_transcribe and input_type == "File" and media_type == "Video" and file:
        (audio_path,), t = timed_step("Извлечение аудио", extract_audio_from_video, file)
        timings.append(("Извлечение аудио", t))
    
    # 2. Транскрипция / OCR
    if do_transcribe:
        if input_type != "File" or not file:
            text = "[Ошибка: нужен файл для транскрипции]"
            timings.append(("Нет файла для транскрипции", 0))
            detected_lang = "unknown"
            detected_prob = 0.0
        else:
            if media_type in ["Audio", "Video"]:
                audio_p = file if media_type == "Audio" else audio_path
                (text, detected_lang, detected_prob), t = timed_step(
                    f"Транскрипция ({media_type})",
                    transcribe_audio, audio_p, source_lang
                )
                timings.append((f"Транскрипция ({media_type})", t))
            elif media_type == "Image":
                # OCR
                def ocr_step():
                    return ocr_image(file, source_lang if source_lang.lower() != "auto" else "en")
                
                text_raw, t_ocr = timed_step("OCR (EasyOCR)", ocr_step)
                
                # fastText определение языка
                def lang_step():
                    return detect_language_fasttext(text_raw)
                
                detected_lang, detected_prob = timed_step("Определение языка (fastText)", lang_step)[0]
                
                text = text_raw
                timings.append(("OCR", t_ocr))
                timings.append(("fastText lang detect", 0.0))  # время уже в timed_step выше
                
                if source_lang.lower() == "auto":
                    source_lang = detected_lang  # используем для перевода
            else:
                text = "[Неверный тип медиа]"
                detected_lang = "unknown"
                detected_prob = 0.0
    else:
        text = input_text if input_type == "Text" and input_text else ""
        detected_lang = "manual"
        detected_prob = 1.0
        timings.append(("Текст взят из поля", 0))
    
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    
    # 3. Перевод
    if do_translate and text.strip():
        translated_text, t = timed_step(f"Перевод → {target_lang}", translate_text, text, source_lang, target_lang, pivot_model)
        timings.append((f"Перевод → {target_lang}", t))
    else:
        translated_text = text
    
    # 4. TTS
    if do_tts and translated_text.strip():
        tts_audio, t = timed_step(f"TTS ({target_lang})", text_to_speech, translated_text, target_lang)
        timings.append((f"TTS ({target_lang})", t))
    
    total_time = time.time() - start_total
    timings.append(("Общее время", total_time))
    
    # Результат
    result = f"Статус моделей:\n{model_status_text}\n\n"
    
    detector_source = "Whisper" if media_type in ["Audio", "Video"] and do_transcribe else "fastText" if media_type == "Image" and do_transcribe else "manual"
    prob_str = f" ({detector_source} вероятность {detected_prob:.0%})" if detected_prob > 0 else ""
    result += f"Оригинальный текст ({detected_lang}{prob_str}):\n{text[:800]}{' ...' if len(text) > 800 else ''}\n\n"
    
    if do_translate:
        result += f"Переведённый текст ({target_lang}):\n{translated_text[:800]}{' ...' if len(translated_text) > 800 else ''}\n\n"
    
    result += "┌──────────────────────────────────────────────────────┐\n"
    result += "│                  Время выполнения                    │\n"
    result += "├──────────────────────────────────────────────────────┤\n"
    for step, sec in timings:
        result += f"│ {step:<48} │ {sec:>6.2f} сек │\n"
    result += "└──────────────────────────────────────────────────────┘\n"
    
    log(f"Завершено за {total_time:.2f} сек")
    log("─" * 60)
    
    return result, tts_audio, tts_audio

# Gradio интерфейс — без изменений (source_lang = "auto" по умолчанию уже есть)
# (копируйте из предыдущей версии, добавьте только комментарий в label)
source_lang = gr.Textbox(
    label="Исходный язык (en, ru, auto — Whisper/fastText автоопределение)",
    value="auto",
    placeholder="auto — Whisper для аудио/видео, fastText после OCR для изображений"
)

# ────────────────────────────────────────────────
# Gradio интерфейс
# ────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), css="""
    .result-textbox textarea {
        min-height: 220px;
        max-height: 65vh;
        overflow-y: auto !important;
        resize: vertical;
        font-family: monospace;
    }
""") as demo:
    
    gr.Markdown("# Local Media Processor (2025–2026 edition)")
    
    with gr.Row():
        whisper_dropdown = gr.Dropdown(choices=list(WHISPER_MODELS.keys()), label="Модель Whisper", value="large-v3")
        tts_dropdown = gr.Dropdown(choices=list(TTS_MODELS.keys()), label="Модель TTS", value="xtts_v2 (multilingual)")
    
    pivot_translation = gr.Dropdown(choices=TRANSLATION_MODELS_PIVOT, label="Модель перевода → EN (pivot)", value="Helsinki-NLP/opus-mt-mul-en")
    
    load_models_btn = gr.Button("Загрузить выбранные модели", variant="primary")
    
    model_status = gr.Textbox(label="Статус моделей", interactive=False, lines=3, value=model_status_text)
    
    with gr.Row():
        input_type = gr.Radio(choices=["File", "Text"], label="Тип входа", value="File")
    
    file_input = gr.File(label="Аудио / Видео / Изображение")
    input_text = gr.Textbox(label="Входной текст", visible=False, lines=5)
    media_type = gr.Dropdown(choices=["Audio", "Video", "Image"], label="Тип медиа", visible=True)
    
    source_lang = gr.Textbox(label="Исходный язык (en, ru, auto...)", value="auto")
    target_lang = gr.Textbox(label="Целевой язык (en, ru, fr...)", value="ru")
    
    with gr.Row():
        do_transcribe = gr.Checkbox(label="Транскрипция / OCR", value=True)
        do_translate   = gr.Checkbox(label="Перевод", value=True)
        do_tts         = gr.Checkbox(label="TTS (речь)", value=True)
    
    process_btn = gr.Button("Обработать", variant="secondary")
    
    output_text = gr.Textbox(
        label="Результат",
        lines=10,
        max_lines=60,
        interactive=False,
        show_copy_button=True,
        elem_classes=["result-textbox"]
    )
    
    output_audio = gr.Audio(label="Сгенерированная речь")
    output_download = gr.File(label="Скачать аудио")
    
    # ─── События ───
    def update_visibility(inp_type):
        file_vis = inp_type == "File"
        text_vis = inp_type == "Text"
        return (
            gr.update(visible=file_vis),
            gr.update(visible=text_vis),
            gr.update(visible=file_vis)
        )
    
    input_type.change(update_visibility, input_type, [file_input, input_text, media_type])
    
    load_models_btn.click(
        load_selected_models,
        inputs=[whisper_dropdown, tts_dropdown],
        outputs=[model_status]
    )
    
    process_btn.click(
        process_media,
        inputs=[
            input_type, file_input, input_text, media_type,
            source_lang, target_lang,
            do_transcribe, do_translate, do_tts,
            whisper_dropdown, tts_dropdown, pivot_translation
        ],
        outputs=[output_text, output_audio, output_download]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
