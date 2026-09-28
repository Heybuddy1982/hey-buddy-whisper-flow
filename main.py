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
import threading
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# NOTE: faster_whisper is imported inside _load_model on purpose — if a
# native dependency is broken (e.g. missing libgomp), a top-level import
# kills uvicorn before it binds and Railway shows an unexplained 502.
# Importing in the loader keeps /health alive to report the real error.

# Minimal logging — no transcript content ever logged
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hey-buddy-whisper")

app = FastAPI(title="Hey Buddy Whisper STT", docs_url=None, redoc_url=None)

# CORS — known app origins are ALWAYS allowed; the env var can only
# ADD origins, never lock the app out. Field-diagnosed 2026-07-21:
# ALLOWED_ORIGINS on Railway was set for an older domain, so the
# browser blocked every request from the live app before sending —
# /debug/recent stayed empty through six app deploys while the app
# looked deaf. An env var must never be able to silently sever the
# app from its own server.
KNOWN_ORIGINS = [
    "https://hey-buddy-canada.lovable.app",
    "https://id-preview--4f4cac9c-9c01-4950-acfb-59e79cbb47ac.lovable.app",
    "https://app.heybuddyapp.ca",
    "https://hey-buddy-activation.vercel.app",  # live app (2026-09); was env-only
    "http://localhost:5173",
]
_env_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if "*" in _env_origins or not _env_origins:
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = sorted(set(KNOWN_ORIGINS + _env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-Session-ID", "X-Hey-Buddy-Key"],
)

# CRASH-PROOF STARTUP (added after a silent Railway crash-loop caused a
# 502 that the app experienced as 'it doesn't listen'). The web server
# ALWAYS binds and /health ALWAYS answers; the model loads in a
# background thread. If loading fails, /health reports the exact error
# instead of the whole container dying. A safety product's server must
# fail loudly and observably, never silently.
# tiny.en default (2026-07-21): /debug/recent showed 60-190s per
# request on base.en under Railway's shared CPU — the session had
# always moved on before the answer arrived. tiny.en runs 3-5x
# faster; with a keyword-driven classifier downstream, speed beats
# marginal accuracy here. Override with WHISPER_MODEL env if needed.
# PINNED for beta (2026-07-21): a leftover WHISPER_MODEL=base env var
# on Railway silently overrode the tiny.en default — the second env
# var today to steer production against the code's intent (see CORS).
# During beta the code decides. Restore env override only with a
# /health field that names the source of the value.
MODEL_SIZE = "tiny.en"

# ------------------------------------------------------------------
# STT GATEWAY (2026-07-22): primary = hosted fast Whisper (Groq
# whisper-large-v3-turbo, ~300ms/utterance) when GROQ_API_KEY is set;
# fallback = local faster-whisper. Field data showed 60-190s local
# inference on shared CPU — unusable for a first-60-seconds product.
# The app contract is unchanged; the key never leaves this server;
# audio passes through transiently on both paths and is never stored.
# Provider processing is flagged PENDING CGO REVIEW in the app repo's
# DECISIONS.md. An unset key degrades to local — adding capability
# via env is allowed, removing it silently is not.
# ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TIMEOUT_S = float(os.getenv("GROQ_TIMEOUT_S", "8"))
CPU_THREADS = int(os.getenv("WHISPER_THREADS", str(os.cpu_count() or 2)))

model = None
model_state = "loading"   # loading | ready | error
model_error = ""


def _load_model():
    global model, model_state, model_error
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper model: {MODEL_SIZE} (int8, {CPU_THREADS} threads)")
        m = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
        # Warm-up inference: the first real request must not pay
        # cold-path costs (kernel compilation, memory setup).
        try:
            import numpy as np
            warm = np.zeros(16000, dtype=np.float32)  # 1s silence
            list(m.transcribe(warm, language="en", beam_size=1)[0])
            logger.info("Warm-up inference complete")
        except Exception as we:
            logger.warning(f"Warm-up skipped: {type(we).__name__}")
        model = m
        model_state = "ready"
        logger.info("Model loaded")
    except Exception as e:
        model_state = "error"
        model_error = f"{type(e).__name__}: {e}"
        logger.error(f"MODEL LOAD FAILED: {model_error}")


threading.Thread(target=_load_model, daemon=True).start()

# Simple API key check — set HB_API_KEY env var in Railway
API_KEY = os.getenv("HB_API_KEY", "").strip()  # strip: paste-safe, matches GROQ_API_KEY handling


class TranscriptResponse(BaseModel):
    transcript: str
    language: str
    session_id: str
    audio_destroyed: bool = True
    # Server-side transcription time — lets us diagnose latency from the
    # phone next time without needing the Railway metrics dashboard.
    processing_ms: int = 0


SERVER_VERSION = "fw-2026-09-28b"  # bump on every deploy-relevant change


# ------------------------------------------------------------------
# DEBUG RING BUFFER — phone-readable request history at /debug/recent.
# Metadata ONLY: sizes, statuses, timings, character counts. Never
# audio, never transcript content (store-nothing rule). Exists so a
# founder on a phone can see what actually arrived without the
# Railway dashboard. In-memory, capped, gone on restart.
# ------------------------------------------------------------------
from collections import deque
from datetime import datetime, timezone

RECENT: deque = deque(maxlen=30)


def record_rx(session: str, nbytes: int, ctype: str, status: str,
              transcript_chars: int = -1, ms: int = -1, err: str = ""):
    RECENT.appendleft({
        "at": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "session": (session or "anonymous")[:8],
        "bytes": nbytes,
        "type": ctype or "",
        "status": status,
        "transcript_chars": transcript_chars,
        "ms": ms,
        "error": err,
    })


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    model_state: str
    model_error: str = ""


@app.get("/health", response_model=HealthResponse)
def health():
    # status stays "ok" whenever the web server is up — Railway's deploy
    # healthcheck keys off a 200 here. model_state tells the real story.
    return HealthResponse(
        status="ok",
        model=MODEL_SIZE,
        version=SERVER_VERSION,
        model_state=model_state,
        model_error=model_error,
    )


@app.get("/debug/recent")
def debug_recent():
    """Last 30 transcribe attempts, newest first. Metadata only."""
    return {"version": SERVER_VERSION, "model_state": model_state,
            "groq_configured": bool(GROQ_API_KEY),
            "last_groq_error": LAST_GROQ_ERROR,
            "requests": list(RECENT)}


def _to_flac_16k_mono(audio_bytes: bytes) -> bytes | None:
    """Re-encode any browser recording to 16 kHz mono FLAC, in memory.

    GROQ 400 FIX (2026-09-28): field log 2026-09-24 showed Groq rejecting the
    phone's raw MediaRecorder WebM with 400, forcing the local fallback
    (18.4s on shared CPU). Browser WebM from MediaRecorder is streamed
    without duration/cue metadata and its codec params vary by device.
    16 kHz mono FLAC is Groq's documented preferred input: lossless for
    speech, smaller upload, one predictable container for every phone.
    PyAV ships with faster-whisper; no disk, no ffmpeg binary.
    Returns None on any failure so the caller sends the original bytes.
    """
    try:
        import av
        out = io.BytesIO()
        with av.open(io.BytesIO(audio_bytes)) as src:
            in_stream = next(s for s in src.streams if s.type == "audio")
            with av.open(out, mode="w", format="flac") as dst:
                o = dst.add_stream("flac", rate=16000)
                o.layout = "mono"
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                for frame in src.decode(in_stream):
                    for rf in resampler.resample(frame):
                        for pkt in o.encode(rf):
                            dst.mux(pkt)
                for rf in resampler.resample(None):
                    for pkt in o.encode(rf):
                        dst.mux(pkt)
                for pkt in o.encode(None):
                    dst.mux(pkt)
        data = out.getvalue()
        return data if len(data) > 100 else None
    except Exception as e:
        logger.warning(f"FLAC re-encode skipped: {type(e).__name__}")
        return None


LAST_GROQ_ERROR = ""  # Groq's own error message (never audio/transcript)


async def _transcribe_via_groq(audio_bytes: bytes, content_type: str) -> str | None:
    """Returns transcript text, or None to signal fallback to local."""
    global LAST_GROQ_ERROR
    if not GROQ_API_KEY:
        return None
    flac = _to_flac_16k_mono(audio_bytes)
    if flac is not None:
        upload = ("audio.flac", flac, "audio/flac")
    else:
        upload = ("audio.webm", audio_bytes, (content_type or "audio/webm").split(";")[0])
    try:
        import httpx
        async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_S) as client:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": upload},
                data={"model": GROQ_MODEL, "language": "en",
                      "response_format": "json", "temperature": "0"},
            )
        if resp.status_code != 200:
            # Groq's error body is its own message about the request
            # (format, size, model) — never our transcript. Keep it so the
            # next failure is diagnosable from the phone via /debug/recent.
            LAST_GROQ_ERROR = f"{resp.status_code} {upload[0]}: {resp.text[:200]}"
            logger.warning(f"Groq STT {LAST_GROQ_ERROR} — falling back to local")
            return None
        LAST_GROQ_ERROR = ""
        return (resp.json().get("text") or "").strip()
    except Exception as e:
        LAST_GROQ_ERROR = f"{type(e).__name__} ({upload[0]})"
        logger.warning(f"Groq STT {LAST_GROQ_ERROR} — falling back to local")
        return None


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(
    audio: UploadFile = File(...),
    x_session_id: str = Header(default="anonymous"),
    x_hey_buddy_key: str = Header(default=""),
):
    # API key check
    if API_KEY and x_hey_buddy_key != API_KEY:
        record_rx(x_session_id, -1, "", "401 bad key")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Model not up yet (or failed) — tell the app plainly instead of
    # hanging. The app's onError path speaks to the user within ~1s.
    if model_state != "ready":
        record_rx(x_session_id, -1, "", f"503 model {model_state}")
        raise HTTPException(status_code=503, detail=f"Model {model_state}")

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

        # Diagnostic: size + container type only — never content.
        logger.info(
            f"RX session={x_session_id[:8]}... bytes={len(audio_bytes)} "
            f"type={audio.content_type} name={audio.filename}"
        )

        if len(audio_bytes) < 100:
            record_rx(x_session_id, len(audio_bytes), audio.content_type,
                      "400 audio too short")
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({len(audio_bytes)} bytes)",
            )

        if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=413, detail="Audio too large")

        # Decode + transcribe fully in memory (PyAV) — no temp file at all,
        # which is stricter than the old /tmp approach. English forced,
        # greedy decode, built-in VAD skips the trailing silence the app's
        # recorder always captures before it stops.
        t0 = time.monotonic()
        engine = "local"
        groq_text = await _transcribe_via_groq(audio_bytes, audio.content_type or "")
        if groq_text is not None:
            engine = "groq"
            transcript_text = groq_text
            detected_language = "en"
        else:
            def _local_infer() -> str:
                segs, _info = model.transcribe(
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
                return " ".join(seg.text.strip() for seg in segs).strip()
            import asyncio
            transcript_text = await asyncio.get_event_loop().run_in_executor(None, _local_infer)
            detected_language = "en"
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Log session ID + timing only — never log transcript content
        logger.info(
            f"Transcribed session={x_session_id[:8]}... lang={detected_language} "
            f"chars={len(transcript_text)} ms={elapsed_ms}"
        )
        record_rx(x_session_id, len(audio_bytes), audio.content_type,
                  f"200 ok ({engine})", transcript_chars=len(transcript_text), ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error session={x_session_id[:8]}...: {type(e).__name__}")
        record_rx(x_session_id, len(audio_bytes) if audio_bytes else 0,
                  audio.content_type, "500 error", err=type(e).__name__)
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
