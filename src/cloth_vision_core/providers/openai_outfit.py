from __future__ import annotations

import json
from typing import Any

from cloth_vision_core.errors import ProviderError
from cloth_vision_core.models import ItemProfile, OutfitCandidate, OutfitExplanation

SYSTEM_PROMPT = """You write concise Korean copy for digital wardrobe outfit cards.
Use only the supplied color, style, season, category, and score facts. Do not invent brands,
materials, weather, or occasions. Write one short sentence for the reason and one short sentence
for the styling tip. Avoid exaggerated certainty."""

OUTFIT_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reason": {"type": "string", "maxLength": 80},
        "stylist_tip": {"type": "string", "maxLength": 100},
    },
    "required": ["reason", "stylist_tip"],
}


class OpenAIOutfitExplanationProvider:
    def __init__(
        self, *, model: str, api_key: str | None = None, client: Any | None = None
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError("OpenAI support requires the 'openai' extra") from exc
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self._client = client

    def explain(self, candidate: OutfitCandidate, items: list[ItemProfile]) -> OutfitExplanation:
        facts = {
            "scores": {
                "overall": candidate.overall_score,
                "color": candidate.color_score,
                "season": candidate.season_score,
                "style": candidate.style_score,
            },
            "items": [
                {
                    "category": item.category.value,
                    "color_hex": item.color_hex,
                    "style_tags": item.style_tags,
                    "season_tags": item.season_tags,
                }
                for item in items
            ],
        }
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=json.dumps(facts, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "outfit_explanation",
                        "strict": True,
                        "schema": OUTFIT_EXPLANATION_SCHEMA,
                    }
                },
            )
            payload = json.loads(response.output_text)
            reason = str(payload["reason"]).strip()
            tip = str(payload["stylist_tip"]).strip()
            if not reason or not tip:
                raise ValueError("empty explanation")
            return OutfitExplanation(reason=reason[:80], stylist_tip=tip[:100])
        except Exception as exc:
            raise ProviderError("OpenAI outfit explanation failed") from exc
