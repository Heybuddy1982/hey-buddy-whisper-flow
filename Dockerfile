FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fix pkg_resources issue and upgrade pip first
RUN pip install --upgrade pip setuptools wheel pkg_resources 2>/dev/null || \
    pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch first (avoids CUDA download)
RUN pip install --no-cache-dir \
    "torch==2.3.0" \
    "torchaudio==2.3.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install app dependencies one by one to isolate failures
RUN pip install --no-cache-dir "fastapi==0.111.0"
RUN pip install --no-cache-dir "uvicorn[standard]==0.30.1"
RUN pip install --no-cache-dir "python-multipart==0.0.9"
RUN pip install --no-cache-dir "pydantic==2.7.1"
RUN pip install --no-cache-dir "numpy==1.26.4"
RUN pip install --no-cache-dir "openai-whisper==20231117"

COPY main.py .

# Pre-download Whisper base model
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
