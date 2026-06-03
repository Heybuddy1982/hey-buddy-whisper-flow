# Hey Buddy — Whisper STT Server

Self-hosted Whisper speech-to-text. Audio is destroyed after transcription.
No audio stored. No logging of transcript content. Zero retention.

## Environment Variables (set in Railway)

| Variable | Description | Default |
|---|---|---|
| `WHISPER_MODEL` | Model size: tiny/base/small | `base` |
| `HB_API_KEY` | Secret key the app sends in `X-Hey-Buddy-Key` header | (empty = no auth) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `*` |

## Endpoints

- `GET /health` — health check
- `POST /transcribe` — transcribe audio file, returns transcript text

## Headers

- `X-Session-ID` — anonymous session ID for logging
- `X-Hey-Buddy-Key` — API key (required if HB_API_KEY env var is set)
