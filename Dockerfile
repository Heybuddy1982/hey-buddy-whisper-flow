FROM python:3.11-slim

# Cache bust: 2026-07-11 — faster-whisper rewrite
# No PyTorch, no openai-whisper, no apt ffmpeg (PyAV wheels bundle FFmpeg).
# Image drops from multi-GB to a few hundred MB — faster Railway builds,
# faster restarts, less RAM.

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pydantic faster-whisper

COPY main.py .

# Pre-download models into the image so the first request never pays the
# model fetch. base.en is the default; tiny.en baked in as the fast
# fallback (swap via WHISPER_MODEL env var, no rebuild needed).
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8'); WhisperModel('tiny.en', device='cpu', compute_type='int8')"

EXPOSE 8000
COPY start.sh .
RUN chmod +x start.sh
CMD ["/bin/bash", "start.sh"]
