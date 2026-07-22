"""Text-to-speech via edge-tts (free, no API key).

Pluggable: implement the same `synthesize` async-generator signature for Piper,
Kokoro, or any other engine and swap it in the orchestrator.
"""
from collections.abc import AsyncIterator

import edge_tts

from app.config import settings


class EdgeTTS:
    def __init__(self):
        self.voice = settings.tts_voice

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield MP3 audio chunks for `text`."""
        if not text.strip():
            return
        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
