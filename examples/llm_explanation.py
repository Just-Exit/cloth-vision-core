import os
from uuid import uuid4

from cloth_vision_core import (
    ExplanationContext,
    ItemProfile,
    LLMExplanationProvider,
    MatchingEngine,
)
from cloth_vision_core.providers import OpenAITextProvider

source = ItemProfile(
    id=uuid4(),
    color_hex="#222222",
    season_tags=["winter"],
    style_tags=["casual"],
)
target = ItemProfile(
    id=uuid4(),
    color_hex="#FFFFFF",
    season_tags=["winter"],
    style_tags=["casual"],
)
match = MatchingEngine().compare(source, target)

text_provider = OpenAITextProvider(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
explanation = LLMExplanationProvider(text_provider).explain(
    ExplanationContext(match=match, source=source, target=target)
)
print(explanation)
