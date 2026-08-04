from __future__ import annotations

import json

from cloth_vision_core.models import ExplanationContext
from cloth_vision_core.ports import TextGenerationProvider

DEFAULT_EXPLANATION_INSTRUCTION = """\
당신은 패션 아이템 매칭 결과를 설명하는 도우미입니다.
입력 JSON에 있는 점수와 사실만 사용해 간결한 한국어 설명을 작성하세요.
점수는 결정론적 엔진이 계산한 권위 있는 값이므로 다시 계산하거나 변경하지 마세요.
입력에 없는 소재, 패턴, 핏, 브랜드 또는 사용자 취향을 추측하지 마세요.
"""


class LLMExplanationProvider:
    """Transforms a deterministic matching result into user-facing text with an LLM."""

    def __init__(
        self,
        text_provider: TextGenerationProvider,
        instruction: str = DEFAULT_EXPLANATION_INSTRUCTION,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self.text_provider = text_provider
        self.instruction = instruction.strip()

    def explain(self, context: ExplanationContext) -> str:
        facts = {
            "match": {
                "overall_score": context.match.overall_score,
                "color_score": context.match.color_score,
                "season_score": context.match.season_score,
                "style_score": context.match.style_score,
                "category_score": context.match.category_score,
                "deterministic_reasons": context.match.reasons,
            },
            "source": {
                "category": context.source.category.value,
                "color_hex": context.source.color_hex,
                "style_tags": context.source.style_tags,
                "season_tags": context.source.season_tags,
            },
            "target": {
                "category": context.target.category.value,
                "color_hex": context.target.color_hex,
                "style_tags": context.target.style_tags,
                "season_tags": context.target.season_tags,
            },
        }
        prompt = (
            f"{self.instruction}\n\n"
            "다음은 검증된 매칭 사실입니다.\n"
            f"{json.dumps(facts, ensure_ascii=False, sort_keys=True)}"
        )
        return self.text_provider.generate(prompt)
