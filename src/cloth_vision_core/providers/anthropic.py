from __future__ import annotations

from typing import Any

from cloth_vision_core.errors import ProviderError


class AnthropicTextProvider:
    """Claude Messages API adapter loaded through the optional ``anthropic`` extra."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 512,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ProviderError(
                    "Anthropic support requires the 'anthropic' extra: "
                    "uv add 'cloth-vision-core[anthropic]'"
                ) from exc
            client = Anthropic(api_key=api_key) if api_key is not None else Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
        except Exception as exc:
            raise ProviderError("Anthropic text generation failed") from exc
        if not text.strip():
            raise ProviderError("Anthropic returned an empty text response")
        return text.strip()
