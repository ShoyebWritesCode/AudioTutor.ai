# Kids English Voice Tutor

A low-latency, real-time voice tutor for children. The pipeline is:

```
Browser mic → WebSocket → Silero VAD (endpointing) → Groq Whisper (STT)
            → ChromaDB (RAG memory) → Groq/Cerebras LLM (streaming)
            → edge-tts (TTS) → Browser speaker
```

Designed around **latency** and **proper voice-activity detection**. The frontend
is intentionally minimal — the focus is a solid backend architecture.

## Tech choices

| Stage | Choice | Why |
|-------|--------|-----|
| VAD   | **Silero VAD** (local, ONNX) | Fast, accurate endpointing — decides when the child stopped talking |
| STT   | **Groq Whisper** `whisper-large-v3-turbo` | Free tier, very fast, OpenAI-compatible |
| LLM   | **Groq** or **Cerebras** (Llama 3.3 70B) | Free, extremely fast inference; one-line switch |
| TTS   | **edge-tts** | Free, no API key, kid-friendly voices; pluggable |
| RAG   | **ChromaDB** (local MiniLM embeddings) | Free local embeddings (Groq/Cerebras have none) |
| API   | **FastAPI + WebSocket** | Simple, streaming-friendly |

## Latency features

- **Streaming end to end** — LLM tokens stream; TTS starts on the first finished
  sentence, so the child hears audio in a few hundred ms.
- **Tunable endpointing** — `VAD_MIN_SILENCE_MS` trades snappiness vs. patience.
- **Barge-in** — the tutor stops talking the moment the child speaks again.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env       # then paste your free GROQ_API_KEY
```

Get a free Groq key: https://console.groq.com/keys
Optional Cerebras key: https://cloud.cerebras.ai/

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 and click **Start talking** (allow the mic).
First launch downloads the small Silero VAD and MiniLM embedding models.

## Switching the LLM provider

In `.env`:

```
LLM_PROVIDER=cerebras        # or groq
CEREBRAS_MODEL=llama-3.3-70b
```

## Project layout

```
app/
  main.py                  FastAPI app + /ws WebSocket
  config.py                Settings (.env)
  pipeline/
    vad.py                 Silero VAD streaming + endpointing
    stt.py                 Groq Whisper
    llm.py                 Groq/Cerebras (OpenAI-compatible) streaming
    tts.py                 edge-tts (pluggable)
    orchestrator.py        State machine: VAD→STT→RAG→LLM→TTS + barge-in
  rag/
    memory.py              Per-child ChromaDB memory
static/
  index.html               Minimal browser test client
```

## Production notes (next steps)

- Swap the browser `ScriptProcessorNode` for an `AudioWorklet`, and stream TTS
  playback via MediaSource for true low-latency output.
- Consider **WebRTC** if you need phone-call-grade latency + echo cancellation.
- Add auth + a real `user_id` per child (currently hard-coded `demo`).
- For self-hosted/offline TTS, drop in **Piper** or **Kokoro** behind `tts.py`.
- Watch Groq/Cerebras free-tier rate limits; add retry/backoff for 429s.
