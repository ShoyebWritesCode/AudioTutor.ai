"""Streaming Silero VAD wrapper.

Feeds incoming PCM16 audio through Silero's VADIterator in fixed windows and
emits ("start", sample) / ("end", sample) events. This is the latency-critical
endpointing component: it decides *when the child has finished speaking* so the
rest of the pipeline (STT -> LLM -> TTS) can fire.
"""
import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

from app.config import settings


class StreamingVAD:
    def __init__(self):
        self.sample_rate = settings.sample_rate
        # onnx=True keeps the model tiny and CPU-fast.
        self.model = load_silero_vad(onnx=True)
        self.iterator = VADIterator(
            self.model,
            threshold=settings.vad_threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=settings.vad_min_silence_ms,
            speech_pad_ms=settings.vad_speech_pad_ms,
        )
        # Silero expects 512-sample windows at 16kHz (256 at 8kHz).
        self.window = 512 if self.sample_rate == 16000 else 256
        self._buf = np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        self.iterator.reset_states()
        self._buf = np.empty(0, dtype=np.float32)

    def feed(self, pcm16: bytes) -> list[tuple[str, int]]:
        """Push raw PCM16 bytes; return a list of (event, sample_index)."""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, audio])

        events: list[tuple[str, int]] = []
        while len(self._buf) >= self.window:
            chunk = self._buf[: self.window]
            self._buf = self._buf[self.window :]
            out = self.iterator(torch.from_numpy(chunk), return_seconds=False)
            if out:
                if "start" in out:
                    events.append(("start", int(out["start"])))
                if "end" in out:
                    events.append(("end", int(out["end"])))
        return events
