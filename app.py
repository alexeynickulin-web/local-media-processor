import gc
import logging
import os
import tempfile
import time
import asyncio
import shutil
import glob
from pathlib import Path

# Сторонние библиотеки
import torch
import gradio as gr
import ffmpeg  # pip install ffmpeg-python
import easyocr
import fasttext
import yt_dlp
import edge_tts
import whisperx

# HuggingFace
from transformers import pipeline

# ==================== НАСТРОЙКИ И ЛОГИРОВАНИЕ ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log", mode='a', encoding='utf-8')]
)
logger = logging.getLogger("BatchProcessor")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "huggingface")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
logger.info(f"Using device: {DEVICE}")

NLLB_LANG_MAP = {
    "en": "eng_Latn", "ru": "rus_Cyrl", "fr": "fra_Latn", "de": "deu_Latn",
    "es": "spa_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan", "it": "ita_Latn"
}

# ==================== МЕНЕДЖЕР МОДЕЛЕЙ ====================

class ModelManager:
    def __init__(self):
        self.models = {}
        self.fasttext_model = None
        self._load_fasttext()

    def _load_fasttext(self):
        path = os.path.join(MODELS_DIR, "lid.176.bin")
        if not os.path.exists(path):
            try:
                import urllib.request
                urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", path)
            except:
                pass
        try:
            self.fasttext_model = fasttext.load_model(path)
        except:
            pass

    def unload_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    def unload_all(self):
        keys = list(self.models.keys())
        for k in keys:
            self.unload_model(k)

    def get_whisper(self, model_key):
        if "whisper" in self.models and self.models["whisper"]["key"] == model_key:
            return self.models["whisper"]["instance"]
        self.unload_model("nllb")
        self.unload_model("ocr")
        
        logger.info(f"Загрузка Whisper: {model_key}")
        model = whisperx.load_model(model_key, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=os.path.join(MODELS_DIR, "whisper"))
        self.models["whisper"] = {"key": model_key, "instance": model}
        return model

    def get_translator(self, model_key):
        if "nllb" in self.models and self.models["nllb"]["key"] == model_key:
            return self.models["nllb"]["instance"]
        self.unload_model("whisper")
        
        logger.info(f"Загрузка NLLB: {model_key}")
        translator = pipeline("translation", model=model_key, device=0 if DEVICE == "cuda" else -1, max_length=512)
        self.models["nllb"] = {"key": model_key, "instance": translator}
        return translator

    def get_ocr(self, languages):
        key = str(sorted(languages))
        if "ocr" in self.models and self.models["ocr"]["key"] == key:
            return self.models["ocr"]["instance"]
        self.unload_model("whisper")
        
        reader = easyocr.Reader(languages, gpu=DEVICE == "cuda", model_storage_directory=os.path.join(MODELS_DIR, "ocr"))
        self.models["ocr"] = {"key": key, "instance": reader}
        return reader

mm = ModelManager()

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def safe_temp_path(suffix):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.close()
    return tf.name

def extract_audio_ffmpeg(video_path):
    try:
        output_path = safe_temp_path(".wav")
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar=16000, vn=None, loglevel="error")
            .run(overwrite_output=True)
        )
        return output_path
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        return None

def detect_lang(text):
    if not text or not mm.fasttext_model: return "en"
    try:
        pred = mm.fasttext_model.predict(text.replace("\n", " "), k=1)
        return pred[0][0].replace("__label__", "")
    except:
        return "en"

# ==================== ЛОГИКА ОБРАБОТКИ (ЯДРО) ====================

def process_single_file(
    file_path, original_name, output_dir,
    model_config, tasks_config
):
    """
    Обрабатывает один файл и сохраняет результаты сразу в output_dir.
    """
    logs = [f"🔹 Начало: {original_name}"]
    base_name = os.path.splitext(original_name)[0]
    
    # 1. Транскрипция
    full_text = ""
    detected_lang = model_config["source_lang"]
    segments = []
    
    # Определяем тип (если это аудио/видео или изображение)
    ext = os.path.splitext(file_path)[1].lower()
    is_image = ext in [".jpg", ".jpeg", ".png", ".bmp"]
    
    if tasks_config["transcribe"]:
        if is_image:
            reader = mm.get_ocr([model_config["source_lang"]] if model_config["source_lang"] != "auto" else ["en", "ru"])
            res = reader.readtext(file_path, detail=0, paragraph=True)
            full_text = " ".join(res)
            detected_lang = detect_lang(full_text)
        else:
            # Audio/Video
            audio_path = extract_audio_ffmpeg(file_path)
            if not audio_path:
                return "❌ Ошибка аудио", []
            
            try:
                whisper = mm.get_whisper(model_config["whisper_model"])
                result = whisper.transcribe(audio_path, batch_size=16, language=None if model_config["source_lang"] == "auto" else model_config["source_lang"])
                detected_lang = result.get("language", "en")
                
                # Align
                try:
                    align_model, metadata = whisperx.load_align_model(language_code=detected_lang, device=DEVICE)
                    aligned_result = whisperx.align(result["segments"], align_model, metadata, audio_path, DEVICE, return_char_alignments=False)
                    segments = aligned_result["segments"]
                    del align_model, metadata
                    torch.cuda.empty_cache()
                except:
                    segments = result["segments"]
                
                full_text = " ".join([s["text"].strip() for s in segments])
            finally:
                if os.path.exists(audio_path): os.remove(audio_path)
    
    # Сохранение оригинала текста
    if output_dir:
        with open(os.path.join(output_dir, f"{base_name}_orig.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

    # 2. Перевод
    translated_text = ""
    translated_segments = []
    
    if tasks_config["translate"] and full_text:
        try:
            translator = mm.get_translator(model_config["nllb_model"])
            src_code = NLLB_LANG_MAP.get(detected_lang, f"{detected_lang}_Latn")
            tgt_code = NLLB_LANG_MAP.get(model_config["target_lang"], f"{model_config["target_lang"]}_Latn")
            
            # Полный текст
            res = translator(full_text[:3000], src_lang=src_code, tgt_lang=tgt_code)
            translated_text = res[0]['translation_text']
            
            # Сегменты
            if segments:
                texts = [s["text"] for s in segments]
                batch_res = translator(texts, src_lang=src_code, tgt_lang=tgt_code, batch_size=16)
                for i, r in enumerate(batch_res):
                    translated_segments.append({
                        "start": segments[i]["start"], "end": segments[i]["end"], "text": r['translation_text']
                    })
            
            logs.append(f"✅ Перевод ({detected_lang}->{model_config['target_lang']})")
        except Exception as e:
            logs.append(f"⚠️ Ошибка перевода: {e}")

    # 3. Сохранение (Текст, SRT, Аудио)
    generated_files = []
    
    if output_dir:
        # Сохранение перевода
        if translated_text:
            out_txt = os.path.join(output_dir, f"{base_name}_{model_config['target_lang']}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(translated_text)
            generated_files.append(out_txt)
        
        # Сохранение SRT
        if tasks_config["srt"] and (translated_segments or segments):
            import pysrt
            subs = pysrt.SubRipFile()
            use_segs = translated_segments if translated_segments else segments
            for i, s in enumerate(use_segs):
                subs.append(pysrt.SubRipItem(i+1, start=pysrt.SubRipTime(seconds=s['start']), end=pysrt.SubRipTime(seconds=s['end']), text=s['text']))
            
            out_srt = os.path.join(output_dir, f"{base_name}.srt")
            subs.save(out_srt, encoding='utf-8')
            generated_files.append(out_srt)
        
        # TTS
        if tasks_config["tts"] and translated_text:
            voice_map = {"ru": "ru-RU-SvetlanaNeural", "en": "en-US-JennyNeural"}
            voice = voice_map.get(model_config["target_lang"], "en-US-JennyNeural")
            
            out_mp3 = os.path.join(output_dir, f"{base_name}_{model_config['target_lang']}.mp3")
            try:
                asyncio.run(edge_tts.Communicate(translated_text, voice).save(out_mp3))
                generated_files.append(out_mp3)
                logs.append("✅ TTS сохранен")
            except Exception as e:
                logs.append(f"⚠️ Ошибка TTS: {e}")

    return "\n".join(logs), generated_files

# ==================== ОЧЕРЕДЬ ЗАДАЧ (BATCH) ====================

def run_batch_process(
    files_list, folder_path, output_path,
    whisper_model, nllb_model,
    src_lang, tgt_lang,
    do_transcribe, do_translate, do_tts, do_srt,
    progress=gr.Progress()
):
    # 1. Сбор файлов
    all_files = []
    
    # Если указана папка ввода
    if folder_path and os.path.isdir(folder_path):
        types = ('*.mp4', '*.mp3', '*.wav', '*.m4a', '*.jpg', '*.png')
        for t in types:
            all_files.extend(glob.glob(os.path.join(folder_path, t)))
    # Иначе берем из списка загрузки
    elif files_list:
        all_files = [f.name for f in files_list]
    
    if not all_files:
        return "❌ Нет файлов для обработки", ""

    # Если папка вывода не указана, создаем временную папку "processed" в папке скрипта
    if not output_path:
        output_path = os.path.join(PROJECT_ROOT, "processed_output")
    
    os.makedirs(output_path, exist_ok=True)
    
    model_config = {
        "whisper_model": whisper_model,
        "nllb_model": nllb_model,
        "source_lang": src_lang,
        "target_lang": tgt_lang
    }
    
    tasks_config = {
        "transcribe": do_transcribe,
        "translate": do_translate,
        "tts": do_tts,
        "srt": do_srt
    }
    
    global_log = []
    total = len(all_files)
    
    start_time = time.time()
    
    for idx, file_path in enumerate(all_files):
        filename = os.path.basename(file_path)
        progress(idx / total, desc=f"Обработка {idx+1}/{total}: {filename}")
        
        try:
            log_str, gen_files = process_single_file(file_path, filename, output_path, model_config, tasks_config)
            global_log.append(f"--- {filename} ---\n{log_str}")
            
            # Очистка памяти после каждого файла
            mm.unload_all()
            
        except Exception as e:
            logger.error(f"Error on {filename}: {e}")
            global_log.append(f"❌ Ошибка на файле {filename}: {e}")
            
    total_time = time.time() - start_time
    summary = f"🎉 Готово! Обработано файлов: {total}.\n📂 Результаты сохранены в: {output_path}\n⏱️ Время: {total_time:.1f} сек"
    
    return "\n\n".join(global_log), summary

# ==================== UI ====================

def build_interface():
    css = ".container {max-width: 900px; margin: auto;}"
    
    with gr.Blocks(theme="soft", css=css, title="Batch Media Processor") as demo:
        gr.Markdown("## ⚡ Пакетная обработка видео/аудио (Batch Processor)")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Выберите файлы")
                files_input = gr.File(label="Перетащите файлы сюда", file_count="multiple")
                folder_input = gr.Textbox(label="ИЛИ укажите путь к папке с файлами", placeholder="C:/Videos/ToTranslate")
                
                gr.Markdown("### 2. Куда сохранить?")
                output_input = gr.Textbox(label="Папка для результатов", placeholder="C:/Videos/Done (оставьте пустым для папки processed_output)")
                
                gr.Markdown("### 3. Настройки")
                with gr.Group():
                    src_lang = gr.Dropdown(["auto", "en", "ru", "fr", "de", "es"], label="Исходный язык", value="auto")
                    tgt_lang = gr.Dropdown(["ru", "en", "de", "es", "fr"], label="Язык перевода", value="ru")
                
                with gr.Accordion("⚙️ Модели", open=False):
                    w_model = gr.Dropdown(["large-v3", "medium", "small"], label="Whisper", value="medium")
                    n_model = gr.Dropdown(["facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B"], label="NLLB", value="facebook/nllb-200-distilled-600M")

                with gr.Row():
                    do_trans = gr.Checkbox(label="Транскрипция", value=True)
                    do_trsl = gr.Checkbox(label="Перевод", value=True)
                    do_tts = gr.Checkbox(label="Озвучка (TTS)", value=True)
                    do_srt = gr.Checkbox(label="Сохранить SRT", value=True)
                
                btn = gr.Button("🚀 Запустить обработку", variant="primary")
            
            with gr.Column():
                result_info = gr.Textbox(label="Итоговый статус", lines=4)
                logs_out = gr.Textbox(label="Подробный лог", lines=20)

        btn.click(
            run_batch_process,
            inputs=[
                files_input, folder_input, output_input,
                w_model, n_model,
                src_lang, tgt_lang,
                do_trans, do_trsl, do_tts, do_srt
            ],
            outputs=[logs_out, result_info]
        )

    return demo

if __name__ == "__main__":
    app = build_interface()
    # allow_flagging="never" отключает лишние кнопки
    app.queue().launch(server_name="0.0.0.0", server_port=7860)