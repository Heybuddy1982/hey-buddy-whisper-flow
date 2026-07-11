"""
HEY BUDDY — Self-Hosted Whisper STT Server (faster-whisper)

Governance rules enforced here:
- Audio received, transcribed, and DESTROYED in the same request
- No audio written to disk at any point (in-memory decode only)
- No logging of audio content or transcripts
- Transcript TTL: exists only in the HTTP response — not stored
- No PII attached to session IDs
- Error state destroys all in-progress data

LATENCY NOTE (2026-07-11): This replaced vanilla openai-whisper, which on
Railway's shared CPU took 10-20s per clip — the "~10-20s wait after the
user stops speaking" bug lived entirely here, not in the app. faster-whisper
runs the same base.en model through CTranslate2 int8: ~4-8x faster on CPU,
no PyTorch dependency. Same API contract — the app needs no changes.

Deploy: Railway
Model: base.en int8 (env WHISPER_MODEL to swap; tiny.en pre-baked as the
fast fallback if base is still too slow on this instance)
"""

import os
import io
import gc
import time
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

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

# Load model once at startup. int8 quantization: same accuracy class,
# fraction of the CPU time. Use every core the instance gives us.
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base.en")
CPU_THREADS = int(os.getenv("WHISPER_THREADS", str(os.cpu_count() or 2)))
logger.info(f"Loading faster-whisper model: {MODEL_SIZE} (int8, {CPU_THREADS} threads)")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
logger.info("Model loaded")

# Simple API key check — set HB_API_KEY env var in Railway
API_KEY = os.getenv("HB_API_KEY", "")


class TranscriptResponse(BaseModel):
    transcript: str
    language: str
    session_id: str
    audio_destroyed: bool = True
    # Server-side transcription time — lets us diagnose latency from the
    # phone next time without needing the Railway metrics dashboard.
    processing_ms: int = 0


SERVER_VERSION = "fw-2026-07-11"  # bump on every deploy-relevant change


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model=MODEL_SIZE, version=SERVER_VERSION)


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(
    audio: UploadFile = File(...),
    x_session_id: str = Header(default="anonymous"),
    x_hey_buddy_key: str = Header(default=""),
):
    # API key check
    if API_KEY and x_hey_buddy_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if audio.content_type not in (
        "audio/webm", "audio/ogg", "audio/wav", "audio/mp4",
        "audio/mpeg", "audio/flac", "application/octet-stream"
    ):
        logger.warning(f"Unexpected content type: {audio.content_type} — proceeding")

    audio_bytes = None
    transcript_text = ""
    detected_language = "en"
    elapsed_ms = 0

    try:
        # Read audio entirely into memory — never touch disk
        audio_bytes = await audio.read()

        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Audio too short")

        if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=413, detail="Audio too large")

        # Decode + transcribe fully in memory (PyAV) — no temp file at all,
        # which is stricter than the old /tmp approach. English forced,
        # greedy decode, built-in VAD skips the trailing silence the app's
        # recorder always captures before it stops.
        t0 = time.monotonic()
        segments, info = model.transcribe(
            io.BytesIO(audio_bytes),
            language="en",
            task="transcribe",
            beam_size=1,             # greedy, deterministic, fast
            temperature=0.0,
            condition_on_previous_text=False,  # stateless per request
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        # segments is a generator — joining consumes it and finishes the work
        transcript_text = " ".join(s.text.strip() for s in segments).strip()
        detected_language = getattr(info, "language", "en") or "en"
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Log session ID + timing only — never log transcript content
        logger.info(
            f"Transcribed session={x_session_id[:8]}... lang={detected_language} "
            f"chars={len(transcript_text)} ms={elapsed_ms}"
        )

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
        processing_ms=elapsed_ms,
    )
