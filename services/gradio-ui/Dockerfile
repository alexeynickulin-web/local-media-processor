FROM python:3.11-slim

WORKDIR /app

# Copy shared package
COPY services/shared /app/services/shared
RUN pip install --no-cache-dir /app/services/shared

# Copy Gradio UI service
COPY services/gradio-ui /app/services/gradio-ui
RUN pip install --no-cache-dir /app/services/gradio-ui

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "-m", "services.gradio-ui.app"]
