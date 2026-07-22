"""LLM wrapper. Groq and Cerebras both expose OpenAI-compatible APIs, so a
single AsyncOpenAI client with a swappable base_url covers both."""
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings


class LLM:
    def __init__(self):
        if settings.llm_provider == "cerebras":
            self.client = AsyncOpenAI(
                api_key=settings.cerebras_api_key,
                base_url=settings.cerebras_base_url,
            )
            self.model = settings.cerebras_model
        else:  # default: groq
            self.client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
            )
            self.model = settings.groq_model

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yield response text token-by-token."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            frequency_penalty=settings.llm_frequency_penalty,
            presence_penalty=settings.llm_presence_penalty,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
