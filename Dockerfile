FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install setuptools first
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only (smaller, faster build, sufficient for Whisper base)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.30.1" \
    openai-whisper==20231117 \
    python-multipart==0.0.9 \
    "pydantic==2.7.1" \
    "numpy==1.26.4"

COPY main.py .

# Pre-download Whisper base model at build time
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
