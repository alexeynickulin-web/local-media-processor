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
import gc # Для очистки памяти
from pyannote.audio import Pipeline # Диаризация


# ==================== НАСТРОЙКА NLLB (Прямой перевод) ====================
# Словарь маппинга языков (NLLB использует FLORES-200 коды)
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
}

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

TRANSLATION_MODELS = [
    "Helsinki-NLP/opus-mt-mul-en",
    "Helsinki-NLP/opus-mt-tc-big-mul-en",
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-1.3B",
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

def detect_language_fasttext(text):
    if not text.strip():
        return "unknown", 0.0
    # fastText ожидает список строк
    prediction = fasttext_model.predict([text], k=1)
    lang_label = prediction[0][0][0]          # '__label__ru'
    prob = prediction[1][0][0]                # вероятность
    lang_code = lang_label.replace('__label__', '')
    return lang_code, prob

def load_whisper(model_key):
    global whisper_model, current_whisper_name
    if current_whisper_name == model_key and whisper_model is not None:
        return f"Whisper уже загружен: {model_key}"
    model_id = WHISPER_MODELS[model_key]
    compute_type = "float16" if device == "cuda" else "int8"
    whisper_model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        download_root=os.path.join(MODELS_DIR, "whisper")
    )
    current_whisper_name = model_key
    return f"Whisper загружен: {model_key}"

def load_tts(model_key):
    global tts_model, current_tts_name
    if current_tts_name == model_key and tts_model is not None:
        return f"TTS уже загружен: {model_key}"
    model_name = TTS_MODELS[model_key]
    tts_model = TTS(model_name=model_name, progress_bar=True).to(device)
    current_tts_name = model_key
    return f"TTS загружен: {model_key}"

def load_selected_models(whisper_model_key, tts_model_key):
    global model_status_text
    status = []
    if whisper_model_key:
        status.append(load_whisper(whisper_model_key))
    if tts_model_key:
        status.append(load_tts(tts_model_key))
    model_status_text = "\n".join(status) if status else "Модели уже загружены или не выбраны"
    return model_status_text

# ────────────────────────────────────────────────
# Функции обработки
# ────────────────────────────────────────────────

def extract_audio_from_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path, source_lang="auto"):
    if whisper_model is None:
        raise ValueError("Whisper не загружен")
    lang_param = None if source_lang.lower() == "auto" else source_lang
    segments, info = whisper_model.transcribe(
        audio_path, language=lang_param, beam_size=5, vad_filter=True
    )
    full_text = " ".join([s.text for s in segments])
    return full_text, info.language, info.language_probability, segments

def ocr_image(image_path, source_lang="en"):
    result = ocr_reader.readtext(image_path, detail=0, paragraph=True, lang_list=[source_lang])
    return " ".join(result)

def translate_text(text, source_lang, target_lang, model_name):
    if not text.strip():
        return ""
    try:
        translator = get_translator(model_name)
        if "nllb" in model_name.lower():
            src_code = NLLB_LANG_MAP.get(source_lang, "eng_Latn")
            tgt_code = NLLB_LANG_MAP.get(target_lang, "eng_Latn")
            result = translator(text, src_lang=src_code, tgt_lang=tgt_code, max_length=1024)
            return result[0]['translation_text']
        else:
            # Pivot логика (как раньше)
            en_text = translator(text, src_lang=source_lang)[0]['translation_text'] if source_lang.lower() != "en" else text
            if target_lang.lower() == "en":
                return en_text
            tgt_model = f"Helsinki-NLP/opus-mt-en-{target_lang}"
            tgt_translator = get_translator(tgt_model)
            return tgt_translator(en_text)[0]['translation_text']
    except Exception as e:
        return f"[Ошибка перевода: {str(e)}]"

def diarize_audio(audio_path, hf_token):
    if not hf_token or not audio_path:
        return []
    
    print("Запуск диаризации...")
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(torch.device(device))
        diarization = pipe(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        return segments
    except Exception as e:
        print(f"Ошибка диаризации: {e}")
        return []

def merge_transcription_and_diarization(whisper_segments, diar_segments):
    if not diar_segments:
        return "\n".join([f"{s.text}" for s in whisper_segments])
    final = []
    for w_seg in whisper_segments:
        start, end = w_seg.start, w_seg.end
        speakers = {}
        for d in diar_segments:
            o_start = max(start, d["start"])
            o_end = min(end, d["end"])
            overlap = max(0, o_end - o_start)
            if overlap > 0:
                speakers[d["speaker"]] = speakers.get(d["speaker"], 0) + overlap
        best = max(speakers, key=speakers.get) if speakers else "Unknown"
        final.append(f"[{best}] ({start:.1f}-{end:.1f}): {w_seg.text}")
    return "\n".join(final)

def text_to_speech(text, target_lang="en", ref_audio=None):
    if tts_model is None or not text.strip():
        return None
    output_path = tempfile.mktemp(suffix=".wav")
    try:
        if ref_audio and os.path.exists(ref_audio):
            tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=ref_audio, language=target_lang)
        else:
            tts_model.tts_to_file(text=text, file_path=output_path, language=target_lang)  # без speaker_wav — дефолт
        return output_path
    except Exception as e:
        print(f"TTS ошибка: {e}")
        return None


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
    whisper_segments = None
    
    log("─" * 60)
    log(f"Обработка: {input_type} | {media_type} | transcribe={do_transcribe} translate={do_translate} tts={do_tts} diar={do_diarization}")

    # Извлечение аудио
    if do_transcribe and input_type == "File" and media_type == "Video" and file:
        audio_path, t = timed_step("Извлечение аудио", extract_audio_from_video, file)
        timings.append(("Извлечение аудио", t))
    
    # Транскрипция / OCR
    if do_transcribe:
        if input_type != "File" or not file:
            text = "[Ошибка: нужен файл для транскрипции]"
            timings.append(("Нет файла для транскрипции", 0))
        else:
            if media_type in ["Audio", "Video"]:
                audio_p = file if media_type == "Audio" else audio_path
                full_text, detected_lang, detected_prob, whisper_segments = timed_step(
                    f"Транскрипция ({media_type})", transcribe_audio, audio_p, source_lang
                )[0]
                text = full_text
                timings.append((f"Транскрипция ({media_type})", 0))  # время уже учтено

                # Диаризация
                if do_diarization and audio_p:
                    diar_segments, t_diar = timed_step(
                        "Диаризация (pyannote)", diarize_audio, audio_p, hf_token
                    )
                    timings.append(("Диаризация", t_diar))
                    if whisper_segments:
                        text = merge_transcription_and_diarization(whisper_segments, diar_segments)
            elif media_type == "Image":
                # OCR + fastText
                                
                text_raw, t_ocr = timed_step("OCR", lambda: ocr_image(file, source_lang if source_lang != "auto" else "en"))
                detected_lang, detected_prob = timed_step("fastText", lambda: detect_language_fasttext(text_raw))[0]
                text = text_raw
                timings.append(("OCR + fastText", t_ocr))

                if source_lang.lower() == "auto":
                    source_lang = detected_lang  # используем для перевода
            else:
                text = "[Неверный тип медиа]"

    else:
        text = input_text or ""
        timings.append(("Текст взят из поля", 0))
    
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Перевод
    if do_translate and text.strip():
        translated_text, t = timed_step(
            f"Перевод ({translation_model} → {target_lang})",
            translate_text, text, source_lang, target_lang, translation_model
        )
        timings.append((f"Перевод", t))
    else:
        translated_text = text
    
    # TTS
    if do_tts and translated_text.strip():
        tts_audio, t = timed_step(
            f"TTS ({target_lang}) {'+ cloning' if ref_audio else ''}",
            text_to_speech, translated_text, target_lang, ref_audio if do_cloning else None
        )
        timings.append(("TTS", t))
    
    # Очистка памяти
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    total_time = time.time() - start_total
    timings.append(("Общее время", total_time))
    
    # Результат
    result = f"Статус моделей:\n{model_status_text}\n\n"
    detector = "Whisper" if media_type in ["Audio", "Video"] else "fastText" if media_type == "Image" else "manual"
    prob_str = f" ({detector} вероятность {detected_prob:.0%})" if detected_prob > 0 else ""
    result += f"Оригинальный текст ({detected_lang}{prob_str}):\n{text[:800]}{'...' if len(text)>800 else ''}\n\n"

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
    
    gr.Markdown("# Local Media Processor")
    
    # 1. Поле для токена HF (нужен для диаризации)
    hf_token_input = gr.Textbox(
        label="HuggingFace Token (для диаризации pyannote)", 
        type="password",
        placeholder="hf_..."
    )

    with gr.Row():
        whisper_dropdown = gr.Dropdown(choices=list(WHISPER_MODELS.keys()), label="Модель Whisper", value="large-v3")
        tts_dropdown = gr.Dropdown(choices=list(TTS_MODELS.keys()), label="Модель TTS", value="xtts_v2 (multilingual)")
    
    # 2. Обновляем выбор моделей перевода, добавляем NLLB
    translation_model_dropdown = gr.Dropdown(
        choices=TRANSLATION_MODELS,
        label="Модель перевода",
        value="facebook/nllb-200-distilled-600M" # NLLB по умолчанию
    )

    # 3. Добавляем загрузку аудио для клонирования
    with gr.Row():
        do_cloning = gr.Checkbox(label="Использовать клонирование голоса", value=False)
        ref_audio_input = gr.Audio(
            label="Образец голоса (Reference Audio)", 
            type="filepath", 
            visible=False
        )
    # 4. Чекбокс для диаризации
    do_diarization = gr.Checkbox(label="Включить диаризацию (разделение по спикерам)", value=False)
    
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

    def toggle_cloning(chk):
        return gr.update(visible=chk)

    do_cloning.change(toggle_cloning, do_cloning, ref_audio_input)

    process_btn.click(
        process_media,
        inputs=[
            input_type, file_input, input_text, media_type,
            source_lang, target_lang,
            do_transcribe, do_translate, do_tts,
            whisper_dropdown, tts_dropdown, 
            translation_model_dropdown, # вместо pivot_translation
            do_diarization,
            hf_token_input,
            ref_audio_input # аудио для клонирования
        ],
        outputs=[output_text, output_audio, output_download]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
