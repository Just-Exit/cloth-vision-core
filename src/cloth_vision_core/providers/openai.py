from __future__ import annotations

from typing import Any

from cloth_vision_core.errors import ProviderError


class OpenAITextProvider:
    """OpenAI Responses API adapter loaded through the optional ``openai`` extra."""

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
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError(
                    "OpenAI support requires the 'openai' extra: uv add 'cloth-vision-core[openai]'"
                ) from exc
            client = OpenAI(api_key=api_key) if api_key is not None else OpenAI()
        self.model = model
        self._client = client

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        try:
            response = self._client.responses.create(model=self.model, input=prompt)
            text = response.output_text
        except Exception as exc:
            raise ProviderError("OpenAI text generation failed") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("OpenAI returned an empty text response")
        return text.strip()
