"""Speech-to-text via Groq's Whisper endpoint (OpenAI-compatible)."""
import io
import wave

from openai import AsyncOpenAI

from app.config import settings


class GroqSTT:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self.model = settings.stt_model

    async def transcribe(self, pcm16: bytes) -> str:
        wav_bytes = self._to_wav(pcm16, settings.sample_rate)
        resp = await self.client.audio.transcriptions.create(
            model=self.model,
            file=("speech.wav", wav_bytes, "audio/wav"),
            language="en",
            temperature=0,
        )
        return (resp.text or "").strip()

    @staticmethod
    def _to_wav(pcm16: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16-bit
            w.setframerate(sample_rate)
            w.writeframes(pcm16)
        return buf.getvalue()
