import gc
import logging
import os
import tempfile
import time
from datetime import datetime
from functools import lru_cache

import easyocr
import fasttext
import gradio as gr
import moviepy as mp
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from transformers.pipelines import pipeline
from TTS.api import TTS

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Загрузка fastText модели
fasttext_model = None
if not os.path.exists(FASTTEXT_MODEL_PATH):
    logger.info("Модель fastText lid.176.bin не найдена. Скачиваем автоматически...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)
    logger.info("fastText модель загружена.")

try:
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

# Инициализация OCR с исправленным списком языков
# Используем коды языков, которые точно поддерживаются easyocr
SUPPORTED_OCR_LANGUAGES = ['en', 'ru', 'fr', 'de', 'es', 'ja']  # Убрали 'zh' из начального списка
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
    # Пробуем инициализировать только с английским
    try:
        ocr_reader = easyocr.Reader(
            ['en'],
            download_enabled=True,
            model_storage_directory=os.path.join(MODELS_DIR, "ocr"),
            user_network_directory=os.path.join(MODELS_DIR, "ocr"),
            gpu=device == "cuda"
        )
        logger.info("OCR модель инициализирована только для английского языка")
    except Exception as e2:
        logger.error(f"Критическая ошибка OCR: {e2}")
        ocr_reader = None

@lru_cache(maxsize=32)
def get_translator(model_name):
    """Кэшированный загрузчик моделей перевода"""
    try:
        # Просто не указываем cache_dir - модели будут сохраняться в стандартную директорию
        # которую мы уже настроили через переменные окружения
        return pipeline(
            "translation",
            model=model_name,
            device=device if device == "cuda" else -1
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки модели перевода {model_name}: {e}")
        raise

def log(msg):
    """Унифицированное логирование"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    logger.info(msg)

def timed_step(step_name, func, *args, **kwargs):
    """Измерение времени выполнения шага"""
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
    """Определение языка текста с помощью fastText"""
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
    """Загрузка модели Whisper"""
    global whisper_model, current_whisper_name
    if current_whisper_name == model_key and whisper_model is not None:
        return f"Whisper уже загружен: {model_key}"
    
    try:
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
    except Exception as e:
        logger.error(f"Ошибка загрузки Whisper {model_key}: {e}")
        return f"Ошибка загрузки Whisper: {str(e)}"

def load_tts(model_key):
    """Загрузка модели TTS"""
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

def load_selected_models(whisper_model_key, tts_model_key):
    """Загрузка выбранных моделей"""
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
    """Очистка памяти"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def extract_audio_from_video(video_path):
    """Извлечение аудио из видео"""
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
    """Транскрипция аудио с помощью Whisper"""
    if whisper_model is None:
        raise ValueError("Whisper не загружен")
    
    try:
        lang_param = None if source_lang.lower() == "auto" else source_lang
        segments, info = whisper_model.transcribe(
            audio_path, 
            language=lang_param, 
            beam_size=5, 
            vad_filter=True
        )
        full_text = " ".join([s.text for s in segments])
        return full_text, info.language, info.language_probability, list(segments)
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")
        raise

def ocr_image(image_path, source_lang="en"):
    """Распознавание текста на изображении"""
    if ocr_reader is None:
        raise ValueError("OCR модель не инициализирована")
    
    try:
        # Определяем, какие языки использовать
        if source_lang == "auto":
            # Используем все поддерживаемые языки
            lang_list = SUPPORTED_OCR_LANGUAGES
        else:
            # Проверяем, поддерживается ли запрошенный язык
            if source_lang in SUPPORTED_OCR_LANGUAGES:
                lang_list = [source_lang]
            else:
                # Если язык не поддерживается, используем английский
                lang_list = ['en']
                logger.warning(f"Язык {source_lang} не поддерживается OCR. Используется английский.")
        
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
    """Перевод текста"""
    if not text.strip():
        return ""
    
    try:
        # Если языки совпадают
        if source_lang == target_lang:
            return text
        
        translator = get_translator(model_name)
        
        # Используем NLLB если выбрана эта модель
        if "nllb" in model_name.lower():
            src_code = NLLB_LANG_MAP.get(source_lang, f"{source_lang}_Latn")
            tgt_code = NLLB_LANG_MAP.get(target_lang, f"{target_lang}_Latn")
            
            # Проверяем, поддерживает ли модель эти языки
            result = translator(text, src_lang=src_code, tgt_lang=tgt_code, max_length=1024)
            return result[0]['translation_text']
        else:
            # Pivot логика через английский
            if source_lang.lower() != "en":
                try:
                    # Пробуем прямую модель
                    direct_model = f"Helsinki-NLP/opus-mt-{source_lang}-en"
                    direct_translator = get_translator(direct_model)
                    en_text = direct_translator(text)[0]['translation_text']
                except:
                    # Fallback на мультиязычную модель
                    en_text = translator(text, src_lang=source_lang)[0]['translation_text']
            else:
                en_text = text
            
            if target_lang.lower() == "en":
                return en_text
            
            # Перевод с английского на целевой язык
            try:
                tgt_model = f"Helsinki-NLP/opus-mt-en-{target_lang}"
                tgt_translator = get_translator(tgt_model)
                return tgt_translator(en_text)[0]['translation_text']
            except:
                # Если модель не найдена, возвращаем английский текст
                return en_text
                
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
        return f"[Ошибка перевода: {str(e)}]"

def diarize_audio(audio_path, hf_token):
    """Диаризация аудио (разделение по спикерам)"""
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
    """Объединение транскрипции и диаризации"""
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
    """Синтез речи из текста с поддержкой разных моделей и спикеров"""
    if tts_model is None or not text.strip():
        return None
    
    output_path = tempfile.mktemp(suffix=".wav")
    model_name = tts_model_name or current_tts_name or "unknown"
    
    try:
        # Определяем спикера
        speaker = None
        if model_name in TTS_SPEAKERS:
            speaker = TTS_SPEAKERS[model_name].get(target_lang, TTS_SPEAKERS[model_name]["default"])
        
        # Определяем, какая модель TTS используется
        model_str = str(tts_model).lower()
        
        # Для XTTS v2
        if "xtts" in model_str:
            if ref_audio and os.path.exists(ref_audio):
                tts_model.tts_to_file(
                    text=text, 
                    file_path=output_path, 
                    speaker_wav=ref_audio, 
                    language=target_lang
                )
            else:
                tts_model.tts_to_file(
                    text=text, 
                    file_path=output_path, 
                    language=target_lang
                )
        
        # Для YourTTS
        elif "your_tts" in model_str and speaker:
            try:
                # Пробуем с указанием спикера и языка
                tts_model.tts_to_file(
                    text=text, 
                    file_path=output_path,
                    # speaker=speaker,
                    language=target_lang
                )
            except Exception as e:
                # Fallback: пробуем без языка
                logger.warning(f"YourTTS ошибка с языком {target_lang}: {e}")
                tts_model.tts_to_file(
                    text=text, 
                    file_path=output_path,
                    speaker=speaker
                )
        
        # Для VITS моделей
        elif "vits" in model_str and speaker:
            try:
                # VITS модели обычно используют speaker_id
                tts_model.tts_to_file(
                    text=text, 
                    file_path=output_path,
                    speaker=speaker
                )
            except Exception as e:
                logger.warning(f"VITS ошибка со спикером {speaker}: {e}")
                # Пробуем без спикера
                tts_model.tts_to_file(text=text, file_path=output_path)
        
        # Для других моделей (tacotron и т.д.)
        else:
            try:
                # Сначала пробуем с языком
                tts_model.tts_to_file(
                    text=text, 
                    file_path=output_path, 
                    language=target_lang
                )
            except (TypeError, KeyError):
                # Если не поддерживает язык, пробуем без
                try:
                    tts_model.tts_to_file(text=text, file_path=output_path)
                except Exception as e:
                    logger.error(f"Ошибка базового TTS: {e}")
                    # Последняя попытка с speaker если есть
                    if speaker:
                        try:
                            tts_model.tts_to_file(
                                text=text, 
                                file_path=output_path,
                                speaker=speaker
                            )
                        except:
                            return None
                    else:
                        return None
        
        return output_path
        
    except Exception as e:
        logger.error(f"Критическая ошибка TTS для модели {model_name}: {e}")
        return None

def validate_inputs(input_type, file, input_text, media_type):
    """Валидация входных данных"""
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
    input_type, file, input_text, media_type, source_lang, target_lang,
    do_transcribe, do_translate, do_tts, whisper_model_key, tts_model_key,
    translation_model, do_diarization=False, hf_token="", ref_audio=None
):
    """Основная функция обработки медиа"""
    
    start_total = time.time()
    timings = []
    warnings = []
    result = ""
    
    try:
        # Валидация
        validation_errors = validate_inputs(input_type, file, input_text, media_type)
        if validation_errors:
            return "\n".join([f"❌ {e}" for e in validation_errors]), None, None
        
        # Проверка моделей
        if do_transcribe and whisper_model is None:
            load_whisper(whisper_model_key)
        if do_tts and tts_model is None:
            load_tts(tts_model_key)
        
        if do_transcribe and whisper_model is None:
            warnings.append("⚠️ Модель Whisper не загружена!")
        if do_tts and tts_model is None:
            warnings.append("⚠️ Модель TTS не загружена!")
        
        if warnings:
            return "\n".join(warnings) + "\n\n" + model_status_text, None, None
        
        # Инициализация переменных
        text = ""
        translated_text = ""
        tts_audio_path = None
        detected_lang = source_lang if source_lang != "auto" else "unknown"
        detected_prob = 0.0
        audio_path = None
        whisper_segments = []
        diar_segments = []
        
        log("─" * 60)
        log(f"Обработка: {input_type} | {media_type} | transcribe={do_transcribe} translate={do_translate} tts={do_tts}")
        
        # Шаг 1: Извлечение аудио из видео
        if do_transcribe and input_type == "File" and media_type == "Video" and file:
            try:
                audio_path, t = timed_step("Извлечение аудио", extract_audio_from_video, file)
                timings.append(("Извлечение аудио", t))
            except Exception as e:
                return f"❌ Ошибка извлечения аудио: {str(e)}", None, None
        
        # Шаг 2: Транскрипция / OCR
        if do_transcribe:
            if input_type != "File" or not file:
                text = "[Ошибка: нужен файл для транскрипции]"
                timings.append(("Нет файла для транскрипции", 0))
            else:
                try:
                    if media_type in ["Audio", "Video"]:
                        audio_p = file if media_type == "Audio" else audio_path
                        if not audio_p or not os.path.exists(audio_p):
                            return f"❌ Аудио файл не найден: {audio_p}", None, None
                        
                        # Транскрипция
                        result_tuple = timed_step(
                            f"Транскрипция ({media_type})", 
                            transcribe_audio, 
                            audio_p, 
                            source_lang
                        )
                        full_text, detected_lang, detected_prob, whisper_segments = result_tuple[0]
                        t_transcribe = result_tuple[1]
                        text = full_text
                        timings.append((f"Транскрипция ({media_type})", t_transcribe))
                        
                        # Диаризация (если включена)
                        if do_diarization and hf_token and audio_p:
                            diar_segments, t_diar = timed_step(
                                "Диаризация", 
                                diarize_audio, 
                                audio_p, 
                                hf_token
                            )
                            timings.append(("Диаризация", t_diar))
                            
                            # Объединение с транскрипцией
                            if whisper_segments and diar_segments:
                                text = merge_transcription_and_diarization(whisper_segments, diar_segments)
                    
                    elif media_type == "Image":
                        # OCR
                        text_raw, t_ocr = timed_step(
                            "OCR", 
                            ocr_image, 
                            file, 
                            source_lang if source_lang != "auto" else "auto"
                        )
                        
                        # Определение языка
                        if fasttext_model and text_raw.strip():
                            detected_lang, detected_prob = timed_step(
                                "Определение языка", 
                                detect_language_fasttext, 
                                text_raw
                            )[0]
                        
                        text = text_raw
                        timings.append(("OCR + определение языка", t_ocr))
                        
                        if source_lang.lower() == "auto":
                            source_lang = detected_lang
                    
                    else:
                        text = "[Неверный тип медиа]"
                
                except Exception as e:
                    return f"❌ Ошибка транскрипции/OCR: {str(e)}", None, None
        
        else:
            # Используем текстовый ввод
            text = input_text or ""
            timings.append(("Текст взят из поля", 0))
        
        # Шаг 3: Перевод
        if do_translate and text.strip() and text != "[Неверный тип медиа]" and not text.startswith("[Ошибка"):
            try:
                # Определяем исходный язык для перевода
                actual_source_lang = source_lang if source_lang != "auto" else detected_lang
                
                translated_text, t_translate = timed_step(
                    f"Перевод ({actual_source_lang} → {target_lang})",
                    translate_text,
                    text,
                    actual_source_lang,
                    target_lang,
                    translation_model
                )
                timings.append((f"Перевод ({actual_source_lang} → {target_lang})", t_translate))
            except Exception as e:
                translated_text = f"[Ошибка перевода: {str(e)}]"
                timings.append(("Ошибка перевода", 0))
        else:
            translated_text = text
        
        # Шаг 4: TTS
        if do_tts and translated_text.strip() and not translated_text.startswith("[Ошибка"):
            try:
                # Определяем язык для TTS
                tts_lang = target_lang if do_translate else detected_lang
                
                tts_audio_path, t_tts = timed_step(
                    f"TTS ({tts_lang})",
                    text_to_speech,
                    translated_text,
                    tts_lang,
                    ref_audio,
                    tts_model_key  # Передаём имя модели
                )
                timings.append(("Синтез речи", t_tts))
            except Exception as e:
                tts_audio_path = None
                timings.append(("Ошибка TTS", 0))
                log(f"Ошибка TTS: {e}")
        
        # Очистка временных файлов
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        # Очистка памяти
        cleanup_memory()
        
        # Подготовка результата
        total_time = time.time() - start_total
        timings.append(("Общее время", total_time))
        
        # Форматирование результата
        result = f"📊 **Статус моделей:**\n{model_status_text}\n\n"
        
        # Определение детектора
        if media_type in ["Audio", "Video"]:
            detector = "Whisper"
        elif media_type == "Image":
            detector = "fastText"
        else:
            detector = "manual"
        
        prob_str = f" (вероятность {detected_prob:.0%})" if detected_prob > 0 else ""
        
        if text and not text.startswith("[") and not text.startswith("❌"):
            result += f"📝 **Оригинальный текст** ({detected_lang}{prob_str}):\n{text[:1000]}{'...' if len(text) > 1000 else ''}\n\n"
        
        if do_translate and translated_text and translated_text != text:
            result += f"🌐 **Переведённый текст** ({target_lang}):\n{translated_text[:1000]}{' ...' if len(translated_text) > 1000 else ''}\n\n"
        
        # Таблица времени выполнения
        result += "```\n"
        result += "┌──────────────────────────────────────────────────────┐\n"
        result += "│                  Время выполнения                    │\n"
        result += "├──────────────────────────────────────────────────────┤\n"
        for step, sec in timings:
            result += f"│ {step:<35} │ {sec:>10.2f} сек │\n"
        result += "└──────────────────────────────────────────────────────┘\n"
        result += "```\n"
        
        log(f"✅ Завершено за {total_time:.2f} сек")
        log("─" * 60)
        
        return result, tts_audio_path, tts_audio_path
    
    except Exception as e:
        logger.error(f"Критическая ошибка в process_media: {e}")
        return f"❌ Критическая ошибка: {str(e)}", None, None


# ==================== GRADIO ИНТЕРФЕЙС ====================

# CSS стили для интерфейса
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

with gr.Blocks() as demo:
    
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
                value="your_tts (multilingual)",  # Измените здесь
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
    
    # Информация о поддерживаемых языках
    with gr.Accordion("ℹ️ Поддерживаемые языки", open=False):
        gr.Markdown("""
        ### Транскрипция (Whisper):
        - Поддерживает более 100 языков автоматически
        
        ### OCR (EasyOCR):
        - Английский (en), Русский (ru), Французский (fr)
        - Немецкий (de), Испанский (es), Японский (ja)
        
        ### Перевод (NLLB):
        - Более 200 языков
        
        ### TTS (XTTS v2):
        - Английский, Русский, Французский, Немецкий
        - Испанский, Итальянский, Португальский, Польский
        - Турецкий, Греческий, Болгарский, Датский
        - Финский, Голландский, Чешский, Венгерский
        - Румынский, Шведский
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
            info="Выберите тип загруженного файла"
        )
    
    # Настройки обработки
    with gr.Accordion("⚙️ Настройки обработки", open=True):
        with gr.Row():
            source_lang = gr.Textbox(
                label="Исходный язык", 
                value="auto",
                placeholder="auto, en, ru, fr, de, es, ja...",
                info="'auto' для автоопределения. Для OCR доступны: en, ru, fr, de, es, ja"
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
    
    # Кнопка обработки
    process_btn = gr.Button("🚀 Обработать", variant="secondary", scale=2)
    
    # Результаты
    with gr.Accordion("📊 Результаты", open=True):
        output_text = gr.Textbox(
            label="Результат обработки",
            lines=15,
            max_lines=60,
            interactive=False,
            buttons=["copy"],
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
    
    # ===== ОБРАБОТЧИКИ СОБЫТИЙ =====
    
    def update_visibility(inp_type):
        """Обновление видимости элементов в зависимости от типа ввода"""
        file_vis = inp_type == "File"
        text_vis = inp_type == "Text"
        return (
            gr.update(visible=file_vis),
            gr.update(visible=text_vis),
            gr.update(visible=file_vis)
        )
    
    def toggle_cloning(chk):
        """Переключение видимости поля для референсного аудио"""
        return gr.update(visible=chk)
    
    # Подписка на события
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
            ref_audio_input
        ],
        outputs=[output_text, output_audio, output_download]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        favicon_path=None,
        show_error=True,
        theme="soft",
        css=css
    )