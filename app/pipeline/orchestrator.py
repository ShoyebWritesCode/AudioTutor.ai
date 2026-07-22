"""Per-connection voice session.

Coordinates the full real-time loop:
    audio in -> VAD endpointing -> STT -> RAG -> LLM (stream) -> TTS -> audio out
with barge-in (interrupt the tutor when the child starts talking again).
"""
import asyncio
from datetime import datetime
from pathlib import Path
import time

from fastapi import WebSocket

from app.config import settings
from app.pipeline.llm import LLM
from app.pipeline.stt import GroqSTT
from app.pipeline.tts import EdgeTTS
from app.pipeline.vad import StreamingVAD
from app.rag.memory import Memory

_SENTENCE_END = ".!?"


class VoiceSession:
    def __init__(
        self,
        ws: WebSocket,
        user_id: str,
        stt: GroqSTT,
        llm: LLM,
        tts: EdgeTTS,
        memory: Memory,
    ):
        self.ws = ws
        self.user_id = user_id
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.memory = memory

        self.vad = StreamingVAD()
        self.history: list[dict] = []  # prior turns (no system prompt)
        self.speech_buf = bytearray()
        self.capturing = False
        self.response_task: asyncio.Task | None = None
        self._cancelled = False  # barge-in flag checked by TTS consumer
        self._session_start = datetime.now()
        self._connect_time = time.time()

    # ---- audio ingestion -------------------------------------------------
    async def handle_audio(self, pcm16: bytes) -> None:
        # Guard: Ignore initial microphone pops and static during connection startup
        if time.time() - self._connect_time < 1.5:
            return

        # Guard: If the tutor is speaking and we aren't capturing voice yet,
        # require a loud voice (above threshold) to trigger a barge-in/interrupt.
        # This prevents room hum, clicks, or speaker echo from self-interrupting.
        tutor_speaking = self.response_task and not self.response_task.done()
        if tutor_speaking and not self.capturing:
            import numpy as np
            samples = np.frombuffer(pcm16, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2)) if len(samples) > 0 else 0
            if rms < settings.barge_in_rms_threshold:
                return

        if self.capturing:
            self.speech_buf.extend(pcm16)

        for event, _sample in self.vad.feed(pcm16):
            if event == "start":
                await self._on_speech_start(pcm16)
            elif event == "end":
                await self._on_speech_end()

    async def _on_speech_start(self, pcm16: bytes) -> None:
        # Barge-in: cancel any in-flight tutor response.
        if self.response_task and not self.response_task.done():
            self._cancelled = True          # TTS consumer drains immediately
            self.response_task.cancel()
            # Yield once so the event loop can inject CancelledError into the
            # task. We must NOT await the task here — doing so would block the
            # WebSocket receive loop (same thread), preventing the cancel from
            # ever propagating (deadlock). sleep(0) is enough.
            await asyncio.sleep(0)
            await self._safe_send_json({"type": "interrupt"})

        self._cancelled = False
        self.capturing = True
        self.speech_buf = bytearray(pcm16)
        await self._safe_send_json({"type": "vad", "speech": True})

    async def _on_speech_end(self) -> None:
        if not self.capturing:
            return
        self.capturing = False
        utterance = bytes(self.speech_buf)
        self.speech_buf = bytearray()
        self.vad.reset()  # Reset the VAD iterator states so future runs start clean
        await self._safe_send_json({"type": "vad", "speech": False})
        
        # Guard: Ignore very short audio under 300ms (9600 bytes at 16kHz 16-bit mono)
        # to avoid clicks, pops, or breathing triggering transcription.
        if len(utterance) < 9600:
            return
            
        self.response_task = asyncio.create_task(self._respond(utterance))

    # ---- response pipeline ----------------------------------------------
    async def _respond(self, audio: bytes) -> None:
        start_time = time.perf_counter()
        try:
            stt_start = time.perf_counter()
            text = await self.stt.transcribe(audio)
            stt_duration = time.perf_counter() - stt_start
            
            if not text:
                return

            # Guard against Whisper silence/noise hallucinations (e.g. "Thank you", "you")
            clean_text = text.lower().strip(".,!? ")
            if clean_text in {"you", "thank you", "thank you for watching", "please subscribe", "thanks for watching", "thank you very much"}:
                import numpy as np
                samples = np.frombuffer(audio, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2)) if len(samples) > 0 else 0
                if rms < 800:  # quiet audio/static
                    print(f"[stt] Ignored silence hallucination: '{text}' (RMS: {rms:.1f})")
                    return

            await self._safe_send_json({"type": "transcript", "role": "user", "text": text})

            context = self.memory.retrieve(self.user_id, text)
            messages = self._build_messages(text, context)

            await self._safe_send_json({"type": "tts_start"})

            # Latency tracking variables
            llm_start = time.perf_counter()
            first_llm_token_time = None
            first_audio_chunk_time = None

            # Sentence queue connects the two concurrent tasks below.
            # None is the sentinel that signals "no more sentences".
            sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
            full_tokens: list[str] = []

            async def _llm_producer() -> None:
                """Stream LLM tokens; push complete sentences into sentence_q."""
                nonlocal first_llm_token_time
                sentence = ""
                async for token in self.llm.stream(messages):
                    if first_llm_token_time is None:
                        first_llm_token_time = time.perf_counter() - llm_start
                    full_tokens.append(token)
                    sentence += token
                    await self._safe_send_json({"type": "llm_token", "text": token})
                    if any(p in token for p in _SENTENCE_END):
                        await sentence_q.put(sentence)
                        sentence = ""
                if sentence.strip():
                    await sentence_q.put(sentence)
                await sentence_q.put(None)  # signal done

            async def _tts_consumer() -> None:
                """Pull sentences from sentence_q and stream audio as they arrive.
                Runs concurrently with _llm_producer — while TTS is synthesising
                sentence N, the LLM keeps generating sentence N+1."""
                nonlocal first_audio_chunk_time
                while True:
                    sentence = await sentence_q.get()
                    if sentence is None:
                        break
                    async for audio_chunk in self.tts.synthesize(sentence):
                        if first_audio_chunk_time is None:
                            first_audio_chunk_time = time.perf_counter() - start_time
                        if self._cancelled:
                            return  # drop remaining chunks immediately on barge-in
                        await self.ws.send_bytes(audio_chunk)

            # Both tasks run at the same time — this is the key latency win.
            await asyncio.gather(_llm_producer(), _tts_consumer())

            await self._safe_send_json({"type": "tts_end"})

            full = "".join(full_tokens)
            await self._safe_send_json(
                {"type": "transcript", "role": "assistant", "text": full}
            )

            # Print latency statistics to terminal
            total_duration = time.perf_counter() - start_time
            stt_dur_str = f"{stt_duration:.2f}s"
            llm_ttft_str = f"{first_llm_token_time:.2f}s" if first_llm_token_time is not None else "N/A"
            tts_ttfa_str = f"{first_audio_chunk_time:.2f}s" if first_audio_chunk_time is not None else "N/A"
            total_dur_str = f"{total_duration:.2f}s"
            print(
                f"[latency] STT: {stt_dur_str} | "
                f"LLM TTFT: {llm_ttft_str} | "
                f"TTS TTFA (end-to-end): {tts_ttfa_str} | "
                f"Total Turn: {total_dur_str}"
            )

            # Persist turn for short-term context and long-term memory.
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": full})
            self.history = self.history[-8:]
            self.memory.add_turn(self.user_id, text, full)

        except asyncio.CancelledError:
            # Don't send tts_end — _on_speech_start already sent 'interrupt'
            # which handles frontend cleanup. Sending tts_end here would
            # trigger endMediaSource() on an already-torn-down source.
            raise
        except Exception as exc:  # surface, don't crash the socket
            await self._safe_send_json({"type": "error", "message": str(exc)})

    def _build_messages(self, user_text: str, context: list[str]) -> list[dict]:
        system = settings.system_prompt
        if context:
            remembered = "\n".join(f"- {c}" for c in context)
            system += "\n\nThings you remember about this child:\n" + remembered
        return (
            [{"role": "system", "content": system}]
            + self.history[-6:]
            + [{"role": "user", "content": user_text}]
        )

    async def _safe_send_json(self, payload: dict) -> None:
        try:
            await self.ws.send_json(payload)
        except Exception:
            pass

    async def generate_report(self) -> Path | None:
        """Summarise the session with the LLM and write a .txt report to disk.
        Called after the WebSocket closes, so it never blocks the live session.
        Returns the Path of the saved file, or None if the session was empty.
        """
        if not self.history:
            return None

        # Build a plain transcript for the LLM to summarise.
        lines = []
        for msg in self.history:
            role = "Child" if msg["role"] == "user" else "Tutor"
            lines.append(f"{role}: {msg['content']}")
        transcript = "\n".join(lines)

        duration_min = (datetime.now() - self._session_start).seconds // 60
        turns = sum(1 for m in self.history if m["role"] == "user")

        prompt = (
            f"You are an expert English tutor supervisor reviewing a tutoring session "
            f"for a young child (ages 5-10). The session lasted approximately "
            f"{duration_min} minute(s) and had {turns} child turn(s).\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"Write a concise session report with these sections:\n"
            f"1. Topics & Vocabulary — main subjects discussed, new words used\n"
            f"2. Strengths — what the child did well\n"
            f"3. Areas to Improve — grammar, pronunciation, or confidence gaps noticed\n"
            f"4. Tutor Notes — suggestions for the next session\n"
            f"Keep the tone warm and constructive. Plain text, no markdown."
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        report_parts: list[str] = []
        async for token in self.llm.stream(messages):
            report_parts.append(token)
        report_text = "".join(report_parts).strip()

        # Save to data/reports/<timestamp>_<user_id>.txt
        reports_dir = Path("data") / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
        report_path = reports_dir / f"{timestamp}_{self.user_id}.txt"

        header = (
            f"SESSION REPORT\n"
            f"User   : {self.user_id}\n"
            f"Date   : {self._session_start.strftime('%Y-%m-%d %H:%M')}\n"
            f"Turns  : {turns}\n"
            f"{'=' * 50}\n\n"
        )
        report_path.write_text(header + report_text, encoding="utf-8")
        return report_path
