FROM python:3.11-slim

WORKDIR /app

# Copy shared package
COPY services/shared /app/services/shared
RUN pip install --no-cache-dir -e /app/services/shared

# Install dependencies directly (not as package)
RUN pip install --no-cache-dir \
    "gradio>=4.0.0" \
    "httpx>=0.26.0" \
    "pydantic>=2.0.0"

# Copy Gradio UI service
COPY services/gradio-ui /app/services/gradio-ui

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "services/gradio-ui/app.py"]
