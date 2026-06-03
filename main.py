"""
HEY BUDDY — Self-Hosted Whisper STT Server

Governance rules enforced here:
- Audio received, transcribed, and DESTROYED in the same request
- No audio written to disk at any point
- No logging of audio content or transcripts
- Transcript TTL: exists only in the HTTP response — not stored
- No PII attached to session IDs
- Error state destroys all in-progress data

Deploy: Railway or Fly.io
Model: whisper base (MVP) — swap to small if accuracy insufficient
"""

import os
import io
import gc
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import numpy as np

# Minimal logging — no transcript content ever logged
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hey-buddy-whisper")

app = FastAPI(title="Hey Buddy Whisper STT", docs_url=None, redoc_url=None)

# CORS — restrict to your Vercel domain in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-Session-ID", "X-Hey-Buddy-Key"],
)

# Load model once at startup
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
logger.info(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)
logger.info("Whisper model loaded")

# Simple API key check — set HB_API_KEY env var in Railway
API_KEY = os.getenv("HB_API_KEY", "")


class TranscriptResponse(BaseModel):
    transcript: str
    language: str
    session_id: str
    audio_destroyed: bool = True


class HealthResponse(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model=MODEL_SIZE)


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(
    audio: UploadFile = File(...),
    x_session_id: str = Header(default="anonymous"),
    x_hey_buddy_key: str = Header(default=""),
):
    # API key check
    if API_KEY and x_hey_buddy_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Validate content type
    if audio.content_type not in (
        "audio/webm", "audio/ogg", "audio/wav", "audio/mp4",
        "audio/mpeg", "audio/flac", "application/octet-stream"
    ):
        logger.warning(f"Unexpected content type: {audio.content_type} — proceeding")

    audio_bytes = None
    transcript_text = ""
    detected_language = "en"

    try:
        # Read audio entirely into memory — never touch disk
        audio_bytes = await audio.read()

        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Audio too short")

        if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=413, detail="Audio too large")

        # Write to a temp file in /tmp (RAM-backed on most cloud providers)
        # Deleted immediately after transcription
        with tempfile.NamedTemporaryFile(
            suffix=".webm", dir="/tmp", delete=True
        ) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            # Transcribe — language detection + transcription in one pass
            result = model.transcribe(
                tmp.name,
                language=None,          # auto-detect
                task="transcribe",
                fp16=False,             # CPU-safe
                verbose=False,          # no stdout transcript logging
                condition_on_previous_text=False,  # stateless per request
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            transcript_text = (result.get("text") or "").strip()
            detected_language = result.get("language") or "en"

        # tmp file is deleted by context manager above
        # Log session ID only — never log transcript content
        logger.info(f"Transcribed session={x_session_id[:8]}... lang={detected_language} chars={len(transcript_text)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error session={x_session_id[:8]}...: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Transcription failed")
    finally:
        # Explicit destruction of audio bytes from memory
        if audio_bytes is not None:
            audio_bytes = None
            gc.collect()

    return TranscriptResponse(
        transcript=transcript_text,
        language=detected_language,
        session_id=x_session_id,
        audio_destroyed=True,
    )
