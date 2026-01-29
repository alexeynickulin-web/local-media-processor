This repository is a local, mostly-monolithic media processing app with early-stage
service scaffolding. The instructions below help an AI coding agent be immediately
productive and consistent with the project's architecture and conventions.

1. Big picture
- **Purpose**: offline media processor (ASR, OCR, translation, TTS) with a Gradio
  web UI. The single-entry implementation is in [app.py](app.py); `services/` holds
  optional microservice scaffolds (asr, ocr, translate, tts, orchestrator).
- **Runtime**: runs in Docker via `docker compose up --build` (see [README.md](README.md)).
- **Models**: downloaded to `models/` (HF cache is set to `models/huggingface`).

2. Key files and components
- `app.py`: primary app, UI (Gradio), and processing pipeline (functions like
  `process_single_file`, `run_batch_process`). Inspect `ModelManager` here for
  model lifecycle (load/unload) and how device/compute types are chosen.
- `models/`: runtime model storage. `app.py` sets `HF_HOME` to `models/huggingface`.
- `shared/schemas/`: Pydantic schemas used by prospective services. Example:
  [shared/schemas/segment.py](shared/schemas/segment.py) defines `Segment`.
- `services/*`: scaffolds for extracting per-stage microservices. Files are
  mostly placeholders — treat them as planned boundaries rather than working
  implementations.

3. Data flow & conventions
- Input: user uploads files (audio/video/image) via Gradio UI in `app.py`.
- Processing order in `process_single_file`: detect input type → ASR/OCR → (optional)
  translation → SRT generation → TTS. Segments are lists of objects with
  `start`, `end`, `text` (see `Segment` schema).
- Model switching: `ModelManager` keeps a small registry (`mm`) and explicitly
  unloads other heavy models when loading a new type (e.g., loading ASR unloads
  translation models). Follow this pattern when adding services to avoid OOMs.

4. Developer workflows (practical commands)
- Build & run (recommended):
  - `docker compose build --no-cache`
  - `docker compose up --build`
  - `docker compose up` (no rebuild)
- Run locally (dev, requires Python env):
  - `python -m pip install -r requirements.txt`
  - `python app.py` (Gradio server on port 7860 by default)
- Cleanup models: remove `models/` to free space (first-run downloads are large).

5. Patterns to follow when contributing
- Keep the single-file processing logic intact if you refactor: preserve
  `process_single_file` signature or provide backwards-compatible adapters.
- When extracting microservices from `app.py` into `services/*`, reuse
  `shared/schemas/` for Pydantic contracts and maintain the same JSON shapes
  (e.g., `ASRResponse` includes `language`, `text`, `segments`).
- Use the `MODELS_DIR` / `HF_HOME` conventions for model storage to avoid
  scattering model files across the system.

6. Integration points & external dependencies
- FFmpeg (via `ffmpeg-python`) for audio extraction — errors here often mean
  missing system FFmpeg. Verify `ffmpeg` is available in PATH inside containers.
- `faster-whisper` / `whisperx` for ASR and alignment. `easyocr` for OCR.
- `transformers` pipeline used for translation (NLLB examples in `app.py`).
- TTS uses `edge_tts` in `app.py` (async save with `asyncio.run`).

7. Debugging tips
- Logs: `app.log` is created by `app.py` (root logger). Check it for stack traces.
- Memory issues: follow the `ModelManager.unload_all()` pattern and call
  `torch.cuda.empty_cache()` when unloading GPU models.
- If a model download stalls, confirm network access inside Docker and enough
  disk space (~15–25 GB for many models).

8. What not to change lightly
- Don't change the `models/` layout or `HF_HOME` location unless you update
  all model-loading code paths in `app.py` and services.
- Avoid breaking the Pydantic schemas in `shared/schemas/` — they are the
  contract for future services and UI code.

If anything in these instructions is unclear or you'd like the agent to add
examples (small refactors, service extraction templates, or CI checks), tell
me which area to expand and I'll iterate.
