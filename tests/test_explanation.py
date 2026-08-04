from types import SimpleNamespace
from uuid import uuid4

import pytest

from cloth_vision_core import (
    Category,
    ExplanationContext,
    ItemProfile,
    LLMExplanationProvider,
    MatchResult,
    ProviderError,
)
from cloth_vision_core.providers import (
    AnthropicTextProvider,
    GeminiTextProvider,
    OpenAITextProvider,
)


class RecordingTextProvider:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "두 아이템은 안정적인 조합입니다."


def test_llm_explanation_uses_structured_deterministic_facts() -> None:
    source = ItemProfile(id=uuid4(), category=Category.TOP, color_hex="#112233")
    target = ItemProfile(id=uuid4(), category=Category.BOTTOM, color_hex="#FFFFFF")
    match = MatchResult(
        source_item_id=source.id,
        target_item_id=target.id,
        overall_score=87,
        color_score=92,
        season_score=80,
        style_score=75,
        category_score=95,
        reasons=["색상 조합이 안정적입니다."],
    )
    text_provider = RecordingTextProvider()

    explanation = LLMExplanationProvider(text_provider).explain(
        ExplanationContext(match=match, source=source, target=target)
    )

    assert explanation == "두 아이템은 안정적인 조합입니다."
    assert '"overall_score": 87' in text_provider.prompt
    assert '"category": "top"' in text_provider.prompt
    assert "다시 계산하거나 변경하지 마세요" in text_provider.prompt


def test_openai_adapter_uses_responses_api() -> None:
    calls = []
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(output_text=" 설명 ")
        )
    )

    result = OpenAITextProvider(model="test-openai", client=client).generate("prompt")

    assert result == "설명"
    assert calls == [{"model": "test-openai", "input": "prompt"}]


def test_anthropic_adapter_uses_messages_api() -> None:
    calls = []
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: (
                calls.append(kwargs)
                or SimpleNamespace(content=[SimpleNamespace(type="text", text="설명")])
            )
        )
    )

    result = AnthropicTextProvider(model="test-claude", client=client).generate("prompt")

    assert result == "설명"
    assert calls[0]["model"] == "test-claude"
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]


def test_gemini_adapter_uses_generate_content() -> None:
    calls = []
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(text="설명")
        )
    )

    result = GeminiTextProvider(model="test-gemini", client=client).generate("prompt")

    assert result == "설명"
    assert calls == [{"model": "test-gemini", "contents": "prompt"}]


def test_provider_errors_are_normalized() -> None:
    def fail(**kwargs):
        del kwargs
        raise RuntimeError("secret provider detail")

    client = SimpleNamespace(responses=SimpleNamespace(create=fail))

    with pytest.raises(ProviderError, match="OpenAI text generation failed"):
        OpenAITextProvider(model="test-openai", client=client).generate("prompt")
