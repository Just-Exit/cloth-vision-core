from __future__ import annotations

from typing import Any

from cloth_vision_core.errors import ProviderError


class GeminiTextProvider:
    """Gemini Developer API adapter loaded through the optional ``gemini`` extra."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderError(
                    "Gemini support requires the 'gemini' extra: uv add 'cloth-vision-core[gemini]'"
                ) from exc
            client = genai.Client(api_key=api_key) if api_key is not None else genai.Client()
        self.model = model
        self._client = client

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt)
            text = response.text
        except Exception as exc:
            raise ProviderError("Gemini text generation failed") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Gemini returned an empty text response")
        return text.strip()
