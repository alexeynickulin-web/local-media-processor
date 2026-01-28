import gc
import logging
import os
import tempfile
import time
from datetime import datetime
from functools import lru_cache
from retrying import retry

import easyocr
import fasttext
import gradio as gr
import moviepy as mp
import torch
from pyannote.audio import Pipeline
from transformers.pipelines import pipeline
from TTS.api import TTS

import yt_dlp  # Для YouTube
from pydub import AudioSegment  # Для VAD
import pysrt  # Для SRT (pip install pysrt)


# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # Улучшено: debug для деталей
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import whisperx
except ImportError:
    logger.error("whisperx не установлен. Установите: pip install git+https://github.com/m-bain/whisperX.git")
    whisper_model = None

# ==================== НАСТРОЙКА NLLB ====================
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
    "nl": "nld_Latn",  # Добавлено: голландский
    # Добавьте больше по необходимости
}

# ==================== НАСТРОЙКА ПУТЕЙ ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for subdir in ["whisper", "tts", "translation", "ocr", "huggingface", "fasttext"]:
    os.makedirs(os.path.join(MODELS_DIR, subdir), exist_ok=True)

os.environ["HF_HUB_CACHE"] = os.path.join(MODELS_DIR, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODELS_DIR, "huggingface")
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "huggingface")
os.environ["COQUI_TTS_CACHE"] = os.path.join(MODELS_DIR, "tts")

FASTTEXT_MODEL_PATH = os.path.join(MODELS_DIR, "fasttext", "lid.176.bin")

logger.info(f"Модели сохраняются в: {MODELS_DIR}")

# Устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Загрузка fastText модели с retry
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_fasttext():
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        logger.info("Модель fastText lid.176.bin не найдена. Скачиваем автоматически...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)
        logger.info("fastText модель загружена.")

fasttext_model = None
try:
    download_fasttext()
    fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    logger.info("fastText модель загружена для language detection.")
except Exception as e:
    logger.error(f"Ошибка загрузки fastText модели: {e}")
    fasttext_model = None

# Доступные модели
WHISPER_MODELS = {
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "medium": "medium",
    "small": "small",
    "distil-large-v3": "Systran/faster-whisper-distil-large-v3",
}

TTS_MODELS = {
    "your_tts (multilingual)": "tts_models/multilingual/multi-dataset/your_tts",
    "en/tacotron2-DDC": "tts_models/en/ljspeech/tacotron2-DDC",
    "ru/vits": "tts_models/ru/multi-dataset/vits",
    "en/vits": "tts_models/en/ljspeech/vits",
    "en/vits-persian": "tts_models/en/vctk/vits",  # Ещё одна английская модель
}

# Карта спикеров для разных моделей и языков
TTS_SPEAKERS = {
    "your_tts (multilingual)": {
        "en": "LJSpeech",
        "ru": "Russian Female",
        "fr": "French Female", 
        "de": "German Female",
        "es": "Spanish Female",
        "it": "Italian Female",
        "pt": "Portuguese Female",
        "default": "LJSpeech"
    },
    "ru/vits": {
        "ru": "Russian Female",
        "default": "Russian Female"
    },
    "en/vits": {
        "en": "p225",  # speaker_id из VCTK
        "default": "p225"
    },
    "en/vits-persian": {
        "en": "p225",
        "default": "p225"
    }
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

# Инициализация OCR с расширенным списком языков
SUPPORTED_OCR_LANGUAGES = ['en', 'ru', 'fr', 'de', 'es', 'ja', 'ch_sim']  # Добавлен китайский
try:
    ocr_reader = easyocr.Reader(
        SUPPORTED_OCR_LANGUAGES,
        download_enabled=True,
        model_storage_directory=os.path.join(MODELS_DIR, "ocr"),
        user_network_directory=os.path.join(MODELS_DIR, "ocr"),
        gpu=device == "cuda"
    )
    logger.info(f"OCR модель инициализирована для языков: {SUPPORTED_OCR_LANGUAGES}")
except Exception as e:
    logger.error(f"Ошибка инициализации OCR: {e}")
    # Fallback на en и ru
    try:
        ocr_reader = easyocr.Reader(
            ['en', 'ru'],
            download_enabled=True,
            model_storage_directory=os.path.join(MODELS_DIR, "ocr"),
            user_network_directory=os.path.join(MODELS_DIR, "ocr"),
            gpu=device == "cuda"
        )
        logger.info("OCR модель инициализирована для en и ru")
    except Exception as e2:
        logger.error(f"Критическая ошибка OCR: {e2}")
        ocr_reader = None

@lru_cache(maxsize=32)
def get_translator(model_name):
    try:
        return pipeline(
            "translation",
            model=model_name,
            device=device if device == "cuda" else -1,
            max_length=2048  # Увеличено
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки модели перевода {model_name}: {e}")
        raise

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    logger.info(msg)

def timed_step(step_name, func, *args, **kwargs):
    start = time.time()
    log(f"Начало: {step_name}")
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        log(f"Завершено: {step_name} → {elapsed:.2f} сек")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        log(f"Ошибка в {step_name}: {e} → {elapsed:.2f} сек")
        raise

def detect_language_fasttext(text):
    if not text.strip() or fasttext_model is None:
        return "unknown", 0.0
    try:
        prediction = fasttext_model.predict([text], k=1)
        lang_label = prediction[0][0][0]
        prob = prediction[1][0][0]
        lang_code = lang_label.replace('__label__', '')
        return lang_code, prob
    except Exception as e:
        logger.error(f"Ошибка определения языка: {e}")
        return "unknown", 0.0

def load_whisper(model_key):
    global whisper_model, current_whisper_name
    if current_whisper_name == model_key and whisper_model is not None:
        return f"WhisperX уже загружен: {model_key}"
    
    try:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device_type == "cuda" else "int8"
        
        whisper_model = whisperx.load_model(
            model_key,                  # "large-v3", "medium", "small", "large-v2" и т.д.
            device=device_type,
            compute_type=compute_type,
            download_root=os.path.join(MODELS_DIR, "whisper")
        )
        current_whisper_name = model_key
        logger.info(f"WhisperX модель загружена: {model_key} на {device_type}")
        return f"WhisperX загружен: {model_key}"
    except Exception as e:
        logger.error(f"Ошибка загрузки WhisperX {model_key}: {e}")
        return f"Ошибка загрузки WhisperX: {str(e)}"

def load_tts(model_key):
    global tts_model, current_tts_name
    if current_tts_name == model_key and tts_model is not None:
        return f"TTS уже загружен: {model_key}"
    
    try:
        model_name = TTS_MODELS[model_key]
        tts_model = TTS(model_name=model_name, progress_bar=True).to(device)
        current_tts_name = model_key
        return f"TTS загружен: {model_key}"
    except Exception as e:
        logger.error(f"Ошибка загрузки TTS {model_key}: {e}")
        return f"Ошибка загрузки TTS: {str(e)}"


def preprocess_audio(audio_path, use_vad=False, use_uvr=False):
    """Предобработка аудио: VAD (удаление тишины) + UVR (разделение вокала)"""
    if use_uvr:
        # UVR: Используем ultimatevocalremover (предполагаем установлен)
        try:
            from uvr import uvr
            vocal_path, _ = uvr(audio_path)  # Возвращает вокал и инструментал
            audio_path = vocal_path
            logger.info("UVR: Вокал отделён")
        except ImportError:
            logger.warning("UVR не установлен, пропускаем")
    
    if use_vad:
        # Silero VAD: Загружаем модель
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (get_speech_timestamps, _, read_audio, _, _) = utils
        
        sampling_rate = 16000  # Silero ожидает 16kHz
        wav = read_audio(audio_path, sampling_rate=sampling_rate)
        speech_timestamps = get_speech_timestamps(wav, model, threshold=0.6)
        
        if not speech_timestamps:
            return audio_path  # Нет речи
        
        # Собираем только speech сегменты
        audio = AudioSegment.from_wav(audio_path)
        speech_audio = AudioSegment.silent(duration=0)
        for ts in speech_timestamps:
            start_ms = ts['start'] * (1000 / sampling_rate)
            end_ms = ts['end'] * (1000 / sampling_rate)
            speech_audio += audio[start_ms:end_ms]
        
        vad_path = tempfile.mktemp(suffix=".wav")
        speech_audio.export(vad_path, format="wav")
        audio_path = vad_path
        logger.info("VAD: Тишина удалена")
    
    return audio_path

def download_youtube(url):
    """Скачивание YouTube видео/аудио"""
    if not url:
        return None
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tempfile.mktemp(suffix=".mp4"),
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info['requested_downloads'][0]['filepath']
    except Exception as e:
        logger.error(f"Ошибка YouTube: {e}")
        return None

def translate_segments(segments, source_lang, target_lang, model_name):
    """Перевод по сегментам"""
    translated_segments = []
    for seg in segments:
        if seg.text.strip():
            trans_text = translate_text(seg.text.strip(), source_lang, target_lang, model_name)
            translated_segments.append({
                'start': seg.start,
                'end': seg.end,
                'text': trans_text
            })
    return translated_segments

def generate_srt(segments, file_path):
    """Генерация SRT"""
    subs = pysrt.SubRipFile()
    for i, seg in enumerate(segments, 1):
        start = pysrt.SubRipTime(seconds=seg['start'])
        end = pysrt.SubRipTime(seconds=seg['end'])
        subs.append(pysrt.SubRipItem(index=i, start=start, end=end, text=seg['text']))
    subs.save(file_path, encoding='utf-8')
    return file_path

def load_selected_models(whisper_model_key, tts_model_key):
    global model_status_text
    status = []
    
    try:
        if whisper_model_key:
            status.append(load_whisper(whisper_model_key))
        if tts_model_key:
            status.append(load_tts(tts_model_key))
        
        model_status_text = "\n".join(status) if status else "Модели уже загружены или не выбраны"
    except Exception as e:
        model_status_text = f"Ошибка загрузки моделей: {str(e)}"
    
    return model_status_text

def cleanup_memory():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def extract_audio_from_video(video_path):
    try:
        video = mp.VideoFileClip(video_path)
        audio_path = tempfile.mktemp(suffix=".wav")
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return audio_path
    except Exception as e:
        logger.error(f"Ошибка извлечения аудио: {e}")
        raise

def transcribe_audio(audio_path, source_lang="auto"):
    if whisper_model is None:
        raise ValueError("WhisperX модель не загружена")
    
    try:
        # 1. Базовая транскрипция
        transcribe_result = whisper_model.transcribe(
            audio_path,
            language=None if source_lang.lower() == "auto" else source_lang,
            batch_size=16,               # подбери под свою видеокарту (8–32)
            chunk_size=30,               # сек — можно увеличить для длинных файлов
            print_progress=True
        )
        
        detected_lang = transcribe_result.get("language", "unknown")
        detected_prob = transcribe_result.get("language_probability", 0.0)
        
        # 2. Выравнивание (word-level timestamps) — очень важно для SRT
        progress(0.4, desc="Выравнивание сегментов (alignment)...")
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device
        )
        
        aligned_result = whisperx.align(
            transcribe_result["segments"],
            align_model,
            metadata,
            audio_path,
            device,
            return_char_alignments=False  # word-level достаточно
        )
        
        segments = aligned_result["segments"]  # уже с 'start', 'end', 'text', 'words'
        
        # Собираем полный текст
        full_text = " ".join(seg["text"] for seg in segments)
        
        # Для совместимости с предыдущим кодом возвращаем list сегментов
        # Каждый сегмент имеет 'start', 'end', 'text'
        compatible_segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in segments if seg.get("text", "").strip()
        ]
        
        return full_text, detected_lang, detected_prob, compatible_segments
    
    except Exception as e:
        logger.error(f"Ошибка в WhisperX: {e}", exc_info=True)
        raise

def ocr_image(image_path, source_lang="en"):
    if ocr_reader is None:
        raise ValueError("OCR модель не инициализирована")
    
    try:
        if source_lang == "auto":
            lang_list = SUPPORTED_OCR_LANGUAGES
        else:
            lang_list = [source_lang] if source_lang in SUPPORTED_OCR_LANGUAGES else ['en']
            if source_lang not in SUPPORTED_OCR_LANGUAGES:
                logger.warning(f"Язык {source_lang} не поддерживается OCR. Используется 'en'.")
        
        result = ocr_reader.readtext(
            image_path, 
            detail=0, 
            paragraph=True, 
            lang_list=lang_list
        )
        return " ".join(result)
    except Exception as e:
        logger.error(f"Ошибка OCR: {e}")
        raise

def translate_text(text, source_lang, target_lang, model_name):
    if not text.strip():
        return ""
    
    try:
        if source_lang == target_lang:
            return text
        
        translator = get_translator(model_name)
        
        if "nllb" in model_name.lower():
            src_code = NLLB_LANG_MAP.get(source_lang, f"{source_lang}_Latn")
            tgt_code = NLLB_LANG_MAP.get(target_lang, f"{target_lang}_Latn")
            result = translator(text, src_lang=src_code, tgt_lang=tgt_code, max_length=2048)  # Увеличено
            return result[0]['translation_text']
        else:
            # Pivot через en
            if source_lang.lower() != "en":
                try:
                    direct_model = f"Helsinki-NLP/opus-mt-{source_lang}-en"
                    direct_translator = get_translator(direct_model)
                    en_text = direct_translator(text)[0]['translation_text']
                except:
                    en_text = translator(text, src_lang=source_lang)[0]['translation_text']
            else:
                en_text = text
            
            if target_lang.lower() == "en":
                return en_text
            
            try:
                tgt_model = f"Helsinki-NLP/opus-mt-en-{target_lang}"
                tgt_translator = get_translator(tgt_model)
                return tgt_translator(en_text)[0]['translation_text']
            except:
                return en_text  # Fallback
                
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
        return f"[Ошибка перевода: {str(e)}]"

def diarize_audio(audio_path, hf_token):
    if not hf_token or not audio_path:
        return []
    
    logger.info("Запуск диаризации...")
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=hf_token
        ).to(torch.device(device))
        
        diarization = pipe(audio_path)
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start, 
                "end": turn.end, 
                "speaker": speaker
            })
        
        return segments
    except Exception as e:
        logger.error(f"Ошибка диаризации: {e}")
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

def text_to_speech(text, target_lang="en", ref_audio=None, tts_model_name=None):
    if tts_model is None or not text.strip():
        return None
    
    output_path = tempfile.mktemp(suffix=".wav")
    model_name = tts_model_name or current_tts_name or "unknown"
    
    try:
        speaker = None
        if model_name in TTS_SPEAKERS:
            speaker = TTS_SPEAKERS[model_name].get(target_lang, TTS_SPEAKERS[model_name]["default"])
        
        model_str = str(tts_model).lower()
        
        if "xtts" in model_str:
            if ref_audio and os.path.exists(ref_audio):
                tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=ref_audio, language=target_lang)
            else:
                tts_model.tts_to_file(text=text, file_path=output_path, language=target_lang)
        
        elif "your_tts" in model_str:
            tts_model.tts_to_file(text=text, file_path=output_path, speaker=speaker, language=target_lang)  # Убрано комментирование
        
        elif "vits" in model_str:
            tts_model.tts_to_file(text=text, file_path=output_path, speaker=speaker)  # Для VITS без языка, если не поддерживается
        
        else:
            # Для Tacotron и других
            try:
                tts_model.tts_to_file(text=text, file_path=output_path, language=target_lang)
            except:
                tts_model.tts_to_file(text=text, file_path=output_path)  # Fallback без языка
        
        return output_path
        
    except Exception as e:
        logger.error(f"Критическая ошибка TTS для модели {model_name}: {e}")
        return None

def auto_detect_media_type(file_path):
    if not file_path:
        return None
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp3', '.wav', '.ogg']:
        return "Audio"
    elif ext in ['.mp4', '.avi', '.mov']:
        return "Video"
    elif ext in ['.jpg', '.png', '.bmp']:
        return "Image"
    return None

def validate_inputs(input_type, file, input_text, media_type):
    errors = []
    
    if input_type == "File":
        if not file:
            errors.append("Файл не выбран")
        elif media_type not in ["Audio", "Video", "Image"]:
            errors.append("Неверный тип медиа")
    elif input_type == "Text":
        if not input_text.strip():
            errors.append("Текст не введён")
    
    return errors

def process_media(
    progress=gr.Progress(),
    input_type=None, file=None, input_text="", youtube_url="",
    media_type=None, source_lang="auto", target_lang="ru",
    do_transcribe=True, do_translate=True, do_tts=True,
    whisper_model_key="large-v3", tts_model_key="your_tts (multilingual)",
    translation_model="facebook/nllb-200-distilled-600M",
    do_diarization=False, hf_token="",
    ref_audio=None, use_vad=False, use_uvr=False,
    use_direct_translate=False, output_srt=True
):
    progress(0, desc="Начало обработки...")
    start_total = time.time()
    timings = []
    warnings = []
    result = ""

    # ─── Валидация и YouTube ───────────────────────────────────────
    if youtube_url.strip():
        progress(0.05, desc="Скачивание с YouTube...")
        file = download_youtube(youtube_url)
        if not file:
            return "❌ Ошибка скачивания YouTube", None, None, None, None
        media_type = "Video"

    if not media_type and file:
        media_type = auto_detect_media_type(file) or "Unknown"

    if do_transcribe and whisper_model is None:
        load_whisper(whisper_model_key)

    if do_transcribe and whisper_model is None:
        warnings.append("WhisperX не загружен")

    if warnings:
        return "\n".join(warnings) + f"\n\n{model_status_text}", None, None, None, None

    # ─── Подготовка аудио ──────────────────────────────────────────
    audio_p = None
    text = ""
    detected_lang = source_lang if source_lang != "auto" else "unknown"
    detected_prob = 0.0
    whisper_segments = []   # теперь от WhisperX

    if do_transcribe and media_type in ("Audio", "Video"):
        progress(0.2, desc="Подготовка аудио...")
        
        if media_type == "Video":
            audio_p = extract_audio_from_video(file)
        else:
            audio_p = file

        if not audio_p or not os.path.exists(audio_p):
            return "❌ Аудио-файл не найден", None, None, None, None

        audio_p = preprocess_audio(audio_p, use_vad=use_vad, use_uvr=use_uvr)

        # Транскрипция + alignment через WhisperX
        progress(0.35, desc="Транскрипция + выравнивание (WhisperX)...")
        try:
            full_text, lang, prob, segments = timed_step(
                "WhisperX (transcribe + align)",
                transcribe_audio,
                audio_p,
                source_lang
            )[0]

            text = full_text
            detected_lang = lang
            detected_prob = prob
            whisper_segments = segments

            timings.append(("WhisperX транскрипция + alignment", timings[-1][1] if timings else 0))

        except Exception as e:
            text = f"[WhisperX ошибка: {str(e)}]"
            logger.error("WhisperX failed", exc_info=True)

    elif do_transcribe and media_type == "Image":
        # OCR остаётся как было
        text, t_ocr = timed_step("OCR", ocr_image, file, source_lang if source_lang != "auto" else "auto")
        if fasttext_model and text.strip():
            detected_lang, detected_prob = detect_language_fasttext(text)
        timings.append(("OCR + lang detect", t_ocr))

    else:
        text = input_text.strip() or "[Текст не введён]"
        timings.append(("Ввод текста", 0))

    # ─── Перевод ───────────────────────────────────────────────────
    translated_text = text
    translated_segments = []

    if do_translate and text.strip() and not text.startswith("["):
        progress(0.65, desc="Перевод...")
        src_lang = source_lang if source_lang != "auto" else detected_lang

        if whisper_segments:
            # Переводим сегменты → сохраняем тайминги
            translated_segments, t_trans = timed_step(
                "Перевод сегментов",
                translate_segments,
                whisper_segments,
                src_lang,
                target_lang,
                translation_model
            )
            translated_text = "\n".join(s["text"] for s in translated_segments)
        else:
            translated_text, t_trans = timed_step(
                "Перевод полного текста",
                translate_text,
                text,
                src_lang,
                target_lang,
                translation_model
            )
        timings.append((f"Перевод → {target_lang}", t_trans))

    # ─── TTS ───────────────────────────────────────────────────────
    tts_audio_path = None
    if do_tts and translated_text.strip() and not translated_text.startswith("["):
        progress(0.8, desc="Синтез речи...")
        tts_lang = target_lang if do_translate else detected_lang
        tts_audio_path, t_tts = timed_step(
            f"TTS ({tts_lang})",
            text_to_speech,
            translated_text,
            tts_lang,
            ref_audio,
            tts_model_key
        )
        timings.append(("TTS", t_tts))

    # ─── Экспорт файлов ────────────────────────────────────────────
    text_file_path = tempfile.mktemp(suffix=".txt") if translated_text.strip() else None
    if text_file_path:
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(translated_text)

    srt_file_path = None
    if output_srt and (translated_segments or whisper_segments):
        srt_file_path = tempfile.mktemp(suffix=".srt")
        segments_for_srt = translated_segments if do_translate and translated_segments else whisper_segments
        try:
            generate_srt(segments_for_srt, srt_file_path)
        except Exception as e:
            logger.error(f"SRT generation failed: {e}")
            srt_file_path = None

    # ─── Результат ─────────────────────────────────────────────────
    progress(0.95, desc="Формирование результата...")
    
    prob_str = f" ({detected_prob:.0%})" if detected_prob > 0 else ""
    result = f"**WhisperX** | Язык: {detected_lang}{prob_str}\n\n"
    if text:
        result += f"Оригинал:\n{text.strip()[:800]}...\n\n"
    if translated_text != text:
        result += f"Перевод ({target_lang}):\n{translated_text.strip()[:800]}...\n\n"

    total_time = time.time() - start_total
    timings.append(("Всего", total_time))

    result += "```\nВремя выполнения:\n"
    for name, sec in timings:
        result += f"{name:.<40} {sec:>6.1f} с\n"
    result += "```"

    # Очистка
    if audio_p and audio_p != file and os.path.exists(audio_p):
        try:
            os.remove(audio_p)
        except:
            pass
    cleanup_memory()

    progress(1.0, desc="Готово!")
    return result, tts_audio_path, tts_audio_path, text_file_path, srt_file_path

# ==================== GRADIO ИНТЕРФЕЙС ====================

css = """
    .result-textbox textarea {
        min-height: 220px;
        max-height: 65vh;
        overflow-y: auto !important;
        resize: vertical;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 14px;
    }
    .warning {
        color: #ff6b00;
        font-weight: bold;
    }
    .success {
        color: #00aa00;
        font-weight: bold;
    }
    .error {
        color: #ff0000;
        font-weight: bold;
    }
    .info-box {
        padding: 10px;
        border-radius: 5px;
        background: #f0f8ff;
        border-left: 4px solid #4a90e2;
        margin: 10px 0;
    }
"""

with gr.Blocks(css=css, theme="soft") as demo:
    
    gr.Markdown("# 🎯 Local Media Processor")
    gr.Markdown("Транскрипция, OCR, перевод и синтез речи в одном приложении")
    
    # Секция аутентификации
    with gr.Accordion("🔐 Настройки аутентификации", open=False):
        hf_token_input = gr.Textbox(
            label="HuggingFace Token (для диаризации pyannote)", 
            type="password",
            placeholder="hf_...",
            info="Токен нужен только для диаризации"
        )
    
    # Секция выбора моделей
    with gr.Accordion("🤖 Выбор моделей", open=True):
        with gr.Row():
            whisper_dropdown = gr.Dropdown(
                choices=list(WHISPER_MODELS.keys()), 
                label="Модель Whisper", 
                value="large-v3",
                info="Для транскрипции аудио/видео"
            )
            tts_dropdown = gr.Dropdown(
                choices=list(TTS_MODELS.keys()), 
                label="Модель TTS", 
                value="your_tts (multilingual)",
                info="Для синтеза речи"
            )
        
        translation_model_dropdown = gr.Dropdown(
            choices=TRANSLATION_MODELS,
            label="Модель перевода",
            value="facebook/nllb-200-distilled-600M",
            info="NLLB поддерживает больше языков"
        )
    
    # Клонирование голоса
    with gr.Accordion("🎤 Клонирование голоса", open=False):
        do_cloning = gr.Checkbox(
            label="Использовать клонирование голоса", 
            value=False,
            info="Требуется референсное аудио"
        )
        ref_audio_input = gr.Audio(
            label="Образец голоса (Reference Audio)", 
            type="filepath", 
            visible=False
        )
    
    # Загрузка моделей
    load_models_btn = gr.Button("🔄 Загрузить выбранные модели", variant="primary")
    model_status = gr.Textbox(
        label="Статус моделей", 
        interactive=False, 
        lines=3, 
        value=model_status_text
    )
    
    # Информация о поддерживаемых языках (обновлено)
    with gr.Accordion("ℹ️ Поддерживаемые языки", open=False):
        gr.Markdown("""
        ### Транскрипция (Whisper):
        - Поддерживает более 100 языков автоматически
        
        ### OCR (EasyOCR):
        - Английский (en), Русский (ru), Французский (fr)
        - Немецкий (de), Испанский (es), Японский (ja), Китайский (ch_sim)
        
        ### Перевод (NLLB):
        - Более 200 языков
        
        ### TTS (YourTTS и др.):
        - Английский, Русский, Французский, Немецкий
        - Испанский, Итальянский, Португальский и др.
        """)
    
    # Входные данные
    with gr.Accordion("📥 Входные данные", open=True):
        input_type = gr.Radio(
            choices=["File", "Text"], 
            label="Тип входа", 
            value="File"
        )
        
        file_input = gr.File(
            label="Аудио / Видео / Изображение",
            file_types=["audio", "video", "image"]
        )
        
        input_text = gr.Textbox(
            label="Входной текст", 
            visible=False, 
            lines=5,
            placeholder="Введите текст для перевода и синтеза речи..."
        )
        
        media_type = gr.Dropdown(
            choices=["Audio", "Video", "Image"], 
            label="Тип медиа", 
            visible=True,
            info="Автоопределение по файлу, но можно выбрать вручную"
        )

        youtube_input = gr.Textbox(label="YouTube URL (опционально)", placeholder="https://www.youtube.com/watch?v=...")

    
    # Настройки обработки
    with gr.Accordion("⚙️ Настройки обработки", open=True):
        with gr.Row():
            source_lang = gr.Textbox(
                label="Исходный язык", 
                value="auto",
                placeholder="auto, en, ru, fr, de, es, ja...",
                info="'auto' для автоопределения. Для OCR: en, ru, fr, de, es, ja, ch_sim"
            )
            target_lang = gr.Textbox(
                label="Целевой язык", 
                value="ru",
                placeholder="en, ru, fr, de, es, ja...",
                info="Язык для перевода и TTS"
            )
        
        with gr.Row():
            do_transcribe = gr.Checkbox(
                label="Транскрипция / OCR", 
                value=True,
                info="Распознавание речи или текста на изображении"
            )
            do_translate = gr.Checkbox(
                label="Перевод", 
                value=True,
                info="Перевод текста на целевой язык"
            )
            do_tts = gr.Checkbox(
                label="TTS (синтез речи)", 
                value=True,
                info="Преобразование текста в речь"
            )
            do_diarization = gr.Checkbox(
                label="Диаризация", 
                value=False,
                info="Разделение по спикерам (требуется HF токен)"
            )
            use_vad = gr.Checkbox(label="VAD (удалить тишину)", value=False)
            use_uvr = gr.Checkbox(label="UVR (отделить вокал)", value=False)
            use_direct_translate = gr.Checkbox(label="Прямой перевод в Whisper (to EN)", value=False)
            output_srt = gr.Checkbox(label="Вывод SRT субтитров", value=True)
    
    # Кнопка обработки
    process_btn = gr.Button("🚀 Обработать", variant="secondary", scale=2)
    
    # Результаты
    with gr.Accordion("📊 Результаты", open=True):
        output_text = gr.Textbox(
            label="Результат обработки",
            lines=15,
            max_lines=60,
            interactive=False,
            elem_classes=["result-textbox"]
        )
        
        with gr.Row():
            output_audio = gr.Audio(
                label="Сгенерированная речь", 
                type="filepath"
            )
            output_download = gr.File(
                label="Скачать аудио",
                file_types=[".wav", ".mp3"]
            )
            text_export = gr.File(
                label="Скачать текст",
                file_types=[".txt"]
            )

    srt_output=gr.File(label="Скачать SRT")
    # ===== ОБРАБОТЧИКИ СОБЫТИЙ =====
    
    def update_visibility(inp_type):
        file_vis = inp_type == "File"
        text_vis = inp_type == "Text"
        return (
            gr.update(visible=file_vis),
            gr.update(visible=text_vis),
            gr.update(visible=file_vis)
        )
    
    def toggle_cloning(chk):
        return gr.update(visible=chk)
    
    input_type.change(
        update_visibility, 
        inputs=[input_type], 
        outputs=[file_input, input_text, media_type]
    )
    
    do_cloning.change(
        toggle_cloning,
        inputs=[do_cloning],
        outputs=[ref_audio_input]
    )
    
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
            whisper_dropdown, tts_dropdown,
            translation_model_dropdown,
            do_diarization,
            hf_token_input,
            ref_audio_input,
            youtube_input,
            use_vad,
            use_uvr,
            use_direct_translate,
            output_srt
        ],
        outputs=[output_text, output_audio, output_download, text_export, srt_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        favicon_path=None,
        show_error=True
    )