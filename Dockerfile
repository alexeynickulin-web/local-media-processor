FROM python:3.10-slim

# Установка системных зависимостей (для GPU добавьте nvidia-cuda, но для универсальности - базовый)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Для GPU: Если у хоста есть NVIDIA GPU, используйте --gpus all в docker-compose
# Добавьте torch с CUDA: RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY app.py .

CMD ["python", "app.py"]
