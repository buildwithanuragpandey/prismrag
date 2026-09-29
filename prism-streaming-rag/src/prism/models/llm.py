"""
PRISM LLM Adapter

Provides a clean, configurable interface for text generation.
Supports:
  - Ollama (local, default)
  - OpenAI / OpenAI-compatible (Azure, Groq, Together, etc.)
  - Anthropic

All credentials come from environment variables — never hardcoded.

Interface:
    LLMAdapter
        .generate(system_prompt: str, user_prompt: str) -> str
        .generate_async(...)  → str  (async)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Abstract Interface
# ──────────────────────────────────────────────────────────────

class BaseLLMAdapter(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...

    async def generate_async(self, system_prompt: str, user_prompt: str) -> str:
        """Default: run sync in executor. Override for native async."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, system_prompt, user_prompt)


# ──────────────────────────────────────────────────────────────
# Ollama Adapter
# ──────────────────────────────────────────────────────────────

class OllamaAdapter(BaseLLMAdapter):
    """
    Local LLM via Ollama (https://ollama.ai).
    Requires Ollama running locally: `ollama serve`

    Environment:
        OLLAMA_BASE_URL: default http://localhost:11434
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            raise


# ──────────────────────────────────────────────────────────────
# OpenAI-Compatible Adapter
# ──────────────────────────────────────────────────────────────

class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI or OpenAI-compatible API adapter.

    Environment:
        OPENAI_API_KEY: required
        OPENAI_BASE_URL: optional (default: https://api.openai.com/v1)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it in your .env file."
            )
        self._api_key = api_key
        self._base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("OpenAI generation failed: %s", exc)
            raise


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def create_llm_adapter(config) -> BaseLLMAdapter:
    """
    Create an LLM adapter from PRISMConfig.models.llm.

    Args:
        config: PRISMConfig instance

    Returns:
        BaseLLMAdapter instance
    """
    llm_cfg = config.models.llm
    provider = llm_cfg.provider.lower()

    if provider == "ollama":
        return OllamaAdapter(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )
    elif provider in ("openai", "azure", "groq", "together"):
        return OpenAIAdapter(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )
    else:
        logger.warning("Unknown LLM provider '%s'. Defaulting to Ollama.", provider)
        return OllamaAdapter(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
        )
