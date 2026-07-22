"""Central configuration, loaded from environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = (
    "You are Buddy, a warm, patient, and playful English tutor for young children "
    "(ages 5-10). You are talking out loud, so keep every reply SHORT — one or two "
    "simple sentences. Use easy words. Be encouraging and never harsh. When the child "
    "makes a mistake, gently model the correct version instead of lecturing. Ask one "
    "small follow-up question to keep the conversation going. Never use emojis or "
    "special symbols, because your words are read aloud."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    groq_api_key: str = ""
    cerebras_api_key: str = ""

    # Endpoints (OpenAI-compatible)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"

    # LLM
    llm_provider: str = "groq"  # "groq" | "cerebras"
    groq_model: str = "llama-3.1-8b-instant"
    cerebras_model: str = "llama3.1-8b"
    llm_temperature: float = 0.6
    llm_max_tokens: int = 150
    llm_frequency_penalty: float = 0.5  # penalises repeated tokens → prevents loops
    llm_presence_penalty: float = 0.3   # encourages new topics

    # STT (always Groq Whisper)
    stt_model: str = "whisper-large-v3-turbo"

    # TTS
    tts_voice: str = "en-US-AnaNeural"

    # Audio
    sample_rate: int = 16000

    # VAD / endpointing
    vad_threshold: float = 0.55
    vad_min_silence_ms: int = 400  # ms of silence before endpointing — lower = snappier
    vad_speech_pad_ms: int = 50   # padding around speech edges
    barge_in_rms_threshold: float = 150.0  # RMS volume threshold (0-32767) to trigger barge-in

    # RAG
    chroma_path: str = "./data/chroma"

    system_prompt: str = DEFAULT_SYSTEM_PROMPT


settings = Settings()
