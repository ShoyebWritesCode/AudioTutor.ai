"""FastAPI entrypoint: serves the test client and the /ws voice endpoint."""
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline.llm import LLM
from app.pipeline.orchestrator import VoiceSession
from app.pipeline.stt import GroqSTT
from app.pipeline.tts import EdgeTTS
from app.rag.memory import Memory

app = FastAPI(title="Kids English Voice Tutor")

# Singletons — cheap to hold, expensive to recreate per request.
stt = GroqSTT()
llm = LLM()
tts = EdgeTTS()
memory = Memory()

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_provider": llm.model}


@app.websocket("/ws")
async def voice_ws(ws: WebSocket, user_id: str = "demo") -> None:
    await ws.accept()
    session = VoiceSession(ws, user_id, stt, llm, tts, memory)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.handle_audio(msg["bytes"])
            elif msg.get("text") is not None:
                # Reserved for control messages (config, reset, etc.)
                pass
    except WebSocketDisconnect:
        pass
    finally:
        if session.response_task and not session.response_task.done():
            session.response_task.cancel()
        # Generate and save a session summary report after disconnect.
        try:
            report_path = await session.generate_report()
            if report_path:
                print(f"[report] Session report saved → {report_path}")
        except Exception as exc:
            print(f"[report] Failed to generate report: {exc}")
