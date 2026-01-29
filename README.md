# Local Media Processor

Локальный оффлайн-инструмент для обработки медиафайлов:  
транскрипция аудио/видео, OCR изображений, перевод текста, синтез речи (TTS) — всё на вашем компьютере, без облачных API.

Работает через удобный веб-интерфейс на **Gradio**, запускается в **Docker** (поддержка GPU и CPU).

[https://github.com/alexeynickulin-web/local-media-processor](https://github.com/alexeynickulin-web/local-media-processor)

## Возможности

- Транскрипция аудио и видео (faster-whisper)
- Распознавание текста с изображений (EasyOCR)
- Перевод текста между многими языками (Helsinki-NLP Opus-MT)
- Синтез речи (Coqui TTS / XTTS-v2 и другие)
- Гибкий выбор: можно выполнять только один этап или всю цепочку
- Выбор моделей для каждого этапа
- Отдельная кнопка загрузки моделей (чтобы не ждать при первом запуске)
- Подробное логирование и тайминги каждого шага
- Поддержка GPU (CUDA) и CPU
- Всё хранится локально в папке `./models`

## Текущие поддерживаемые модели (2026)

| Этап              | Рекомендуемая модель                  | Альтернативы                              | Примечание                          |
|-------------------|---------------------------------------|-------------------------------------------|-------------------------------------|
| Транскрипция      | large-v3                              | large-v3-turbo, medium, small, distil-large-v3 | faster-whisper                     |
| Перевод (pivot)   | opus-mt-mul-en                        | opus-mt-tc-big-mul-en                     | → английский как промежуточный     |
| TTS               | xtts_v2 (multilingual)                | your_tts, en/tacotron2-DDC, ru/vits       | Coqui TTS                          |
| OCR               | EasyOCR (многоязычный)                | —                                         | Автоматически скачивает шрифты     |

## Требования

- Docker + Docker Compose
- NVIDIA GPU + драйверы + CUDA toolkit (опционально, для ускорения)
- ≥ 8 ГБ RAM (рекомендуется 16+ ГБ)
- Свободное место на диске: ~15–25 ГБ на модели (первый запуск)

## Быстрый старт

1. Склонируйте репозиторий

```bash
git clone https://github.com/alexeynickulin-web/local-media-processor.git
cd local-media-processor
```

2. (Опционально) Создайте .env для настройки порта или других параметров

```Bash
# .env
PORT=7860
```

3. Запустите

```Bash
docker compose up --build
```

4. Откройте в браузере:
http://localhost:7860

5. Нажмите «Загрузить выбранные модели» (первый раз может занять 5–30 минут в зависимости от интернета и железа)

6. Загружайте файлы и пробуйте разные комбинации действий

# Структура проекта
```text
local-media-processor/
├── app.py                  # монолитный Gradio-интерфейс (опционально)
├── docker-compose.yml      # микросервисы + оркестратор
├── data/                   # общий объём для входных/выходных файлов (создаётся при запуске)
├── models/                 # скачанные модели (ASR, OCR, translation и т.д.)
├── services/
│   ├── orchestrator/       # API: /process — запускает цепочку ASR/OCR → translate → TTS
│   ├── asr/                # транскрипция (WhisperX)
│   ├── ocr/                # распознавание текста (EasyOCR)
│   ├── translate/          # перевод (NLLB)
│   └── tts/                # синтез речи (Edge TTS)
├── shared/ml_contracts/    # общие Pydantic-схемы для сервисов
└── README.md
```

## Запуск микросервисов (Docker Compose)

1. Из корня репозитория:
```bash
docker compose up --build
```
2. Оркестратор: `http://localhost:8000`
   - `POST /process` — тело: `{"file_path": "/data/input/audio.wav"}` или `{"file_path": "/data/input/image.png", "file_type": "image"}`
   - Файлы должны лежать в смонтированном каталоге `./data` (например, `./data/input/`).
3. Для GPU (ASR): нужны NVIDIA Container Toolkit и `deploy.resources.reservations.devices` в `docker-compose.yml` (уже настроено).

# Полезные команды

```Bash
# Пересобрать контейнер после изменения кода
docker compose build --no-cache

# Запустить без пересборки (быстрее)
docker compose up

# Остановить
docker compose down

# Очистить модели (если нужно освободить место)
rm -rf models/
```

# Планы на развитие (roadmap)

- Режим реального времени (микрофон → live-транскрипция + перевод + TTS)
- Голосовое клонирование (загрузка референсного аудио 5–10 сек)
- Экспорт в .srt / .vtt с таймкодами
- Batch-обработка нескольких файлов
- Прогресс-бары и возможность отмены длинных задач
- Поддержка более современных моделей (SeamlessM4T v2, MeloTTS, Canary-1b и т.д.)
- Квантизация моделей для слабого железа (GGUF / int8 / int4)
- Диалоговый режим с маленьким LLM для исправления и суммаризации
