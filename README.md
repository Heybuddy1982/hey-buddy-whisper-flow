# Hey Buddy — Voice server (fallback transcription)

**Role since 2026-09-28:** fallback only. The app transcribes live with the
phone's speech service (Android/desktop Chrome). This server is used on iOS,
or when the phone's speech service fails. See the app repo's DECISIONS.md.

Path: app → Supabase Edge Function `whisper-proxy` (holds the key) → this server
`/transcribe` → Groq `whisper-large-v3-turbo` (audio re-encoded to 16 kHz mono
FLAC in memory) → local faster-whisper `tiny.en` if Groq fails.

Audio is never written to disk. No transcript content is logged.

## Environment variables (Railway)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Enables the fast Groq path. Unset = local only (slow). |
| `GROQ_STT_MODEL` | Optional. Default `whisper-large-v3-turbo`. |
| `GROQ_TIMEOUT_S` | Optional. Default `8`. |
| `HB_API_KEY` | Key the proxy sends in `X-Hey-Buddy-Key`. |
| `ALLOWED_ORIGINS` | Optional extra CORS origins (known app origins are always allowed). |

`WHISPER_MODEL` is ignored during beta — the model is pinned to `tiny.en` in code.

## Endpoints

- `GET /health` — status, model state, server version
- `GET /debug/recent` — last 30 attempts (metadata only) + `last_groq_error`
- `POST /transcribe` — multipart `audio`; headers `X-Session-ID`, `X-Hey-Buddy-Key`
