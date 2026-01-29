FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv (Linux installer puts binary in ~/.local/bin)
ADD https://astral.sh/uv/install.sh /uv-install.sh
RUN sh /uv-install.sh && rm /uv-install.sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

WORKDIR /app

# Контракты
COPY shared /contracts

# uv deps (сначала для кэша)
COPY services/asr/pyproject.toml services/asr/uv.lock ./
RUN uv pip install -r pyproject.toml --system --no-cache

# Код
COPY services/asr/app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
