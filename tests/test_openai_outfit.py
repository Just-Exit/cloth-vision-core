import json
from types import SimpleNamespace
from uuid import uuid4

from cloth_vision_core import Category, ItemProfile, OutfitCandidate
from cloth_vision_core.providers import OpenAIOutfitExplanationProvider


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "reason": "차분한 색 조합이 자연스럽습니다.",
                    "stylist_tip": "소품은 간결하게 더해보세요.",
                }
            )
        )


def test_generates_structured_concise_outfit_copy() -> None:
    responses = FakeResponses()
    provider = OpenAIOutfitExplanationProvider(
        model="test-model", client=SimpleNamespace(responses=responses)
    )
    top = ItemProfile(uuid4(), Category.TOP, "#111111", ["casual"], ["fall"])
    bottom = ItemProfile(uuid4(), Category.BOTTOM, "#EEEEEE", ["casual"], ["fall"])
    candidate = OutfitCandidate((top.id, bottom.id), 91, 92, 90, 91, [])

    result = provider.explain(candidate, [top, bottom])

    assert result.reason == "차분한 색 조합이 자연스럽습니다."
    assert responses.kwargs["text"]["format"]["strict"] is True
    assert responses.kwargs["text"]["format"]["schema"]["additionalProperties"] is False
