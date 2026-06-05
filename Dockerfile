FROM python:3.11-slim

# Cache bust: 2026-06-05
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Fix pkg_resources before anything else
RUN pip install --upgrade pip setuptools wheel

# Step 2: CPU-only PyTorch
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 3: App dependencies
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pydantic numpy

# Step 4: Whisper
RUN pip install --no-cache-dir openai-whisper

# Step 5: Copy app
COPY main.py .

# Step 6: Pre-download model
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000
COPY start.sh .
RUN chmod +x start.sh
CMD ["/bin/bash", "start.sh"]
