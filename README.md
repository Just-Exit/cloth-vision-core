# Cloth Vision Core

Provider-neutral Python toolkit for fashion image validation, analysis pipelines, and
explainable item matching.

The package deliberately excludes web frameworks, databases, authentication, and file
storage. Applications provide their own Vision, segmentation, and explanation providers
through small protocols.

## Features

- Image validation, EXIF orientation handling, and representative color extraction
- Provider-neutral Vision analysis pipeline
- Segmentation, explanation, and text-generation provider contracts
- Optional OpenAI, Anthropic Claude, and Google Gemini text adapters
- Deterministic offline mock provider
- JSON-configurable color, season, style, and category matching
- Typed public Python API

## Install

```bash
uv add cloth-vision-core
```

For local development:

```bash
uv sync --extra dev
make check
```

## Image analysis

The default pipeline validates the image and calculates its representative color. Category
and fashion attributes remain unknown until a Vision provider is supplied.

```python
from pathlib import Path

from cloth_vision_core import AnalysisPipeline, PillowImageProcessor

pipeline = AnalysisPipeline(PillowImageProcessor())
result = pipeline.analyze(Path("item.jpg"))

print(result.color_hex)
print(result.category)
```

A provider can be injected without changing the pipeline:

```python
from cloth_vision_core import AnalysisPipeline, PillowImageProcessor
from cloth_vision_core.providers import MockVisionProvider

pipeline = AnalysisPipeline(
    image_processor=PillowImageProcessor(),
    vision_provider=MockVisionProvider(),
)
```

Implement `VisionProvider` or `SegmentationProvider` to connect an external vision model.
Provider SDKs remain isolated in optional adapters rather than the core pipeline.

## LLM explanations

Install only the provider SDK that the application needs:

```bash
uv add 'cloth-vision-core[openai]'
uv add 'cloth-vision-core[anthropic]'
uv add 'cloth-vision-core[gemini]'

# Or install every bundled LLM adapter.
uv add 'cloth-vision-core[llm]'
```

Each adapter accepts an API key directly. If `api_key` is omitted, the official SDK can read
its standard environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
`GEMINI_API_KEY`/`GOOGLE_API_KEY`). The model name is deliberately explicit so applications
can choose and upgrade models without a Core release.

```python
import os
from uuid import uuid4

from cloth_vision_core import (
    ExplanationContext,
    ItemProfile,
    LLMExplanationProvider,
    MatchingEngine,
)
from cloth_vision_core.providers import OpenAITextProvider

source = ItemProfile(id=uuid4(), color_hex="#222222")
target = ItemProfile(id=uuid4(), color_hex="#FFFFFF")
match = MatchingEngine().compare(source, target)

text_provider = OpenAITextProvider(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
explainer = LLMExplanationProvider(text_provider)
explanation = explainer.explain(ExplanationContext(match=match, source=source, target=target))
```

Switching providers only changes the adapter construction:

```python
from cloth_vision_core.providers import AnthropicTextProvider, GeminiTextProvider

claude = AnthropicTextProvider(
    model=os.environ["ANTHROPIC_MODEL"],
    api_key=os.environ["ANTHROPIC_API_KEY"],
)
gemini = GeminiTextProvider(
    model=os.environ["GEMINI_MODEL"],
    api_key=os.environ["GEMINI_API_KEY"],
)
```

The LLM receives the already calculated score breakdown and deterministic reasons. It only
turns those facts into user-facing text; it does not calculate the authoritative score.

## Item matching

```python
from uuid import uuid4

from cloth_vision_core import ItemProfile, MatchingEngine

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
print(match.overall_score)
print(match.reasons)
```

### JSON scoring configuration

`MatchingEngine.from_json()` validates and applies a versioned JSON file. Correlation lookup
is symmetric, so defining `red -> blue` also covers `blue -> red`.

```python
from cloth_vision_core import MatchingEngine

engine = MatchingEngine.from_json("matching.json")
match = engine.compare(source, target)
```

The complete schema is available in
`src/cloth_vision_core/data/default_matching.json`. A custom file can populate correlation
tables without changing Python code:

```json
{
  "version": 1,
  "overall_weights": {
    "color": 0.25,
    "season": 0.25,
    "style": 0.25,
    "category": 0.25
  },
  "color": {
    "correlations": {"red": {"blue": 90}},
    "missing_score": 60,
    "neutral_score": 92,
    "minimum_score": 55
  },
  "season": {
    "correlations": {"winter": {"spring": 80}},
    "missing_score": 60,
    "default_score": 60,
    "same_score": 100
  },
  "style": {
    "correlations": {"casual": {"formal": 75}},
    "default_score": 60,
    "base_match_score": 70,
    "per_shared_tag": 15,
    "maximum_score": 100
  },
  "category": {
    "correlations": {"top": {"bottom": 95}},
    "missing_score": 60,
    "default_score": 60,
    "same_score": 100
  }
}
```

Rules are strict: scores must be integers from 0 to 100, weights must be from 0 to 1 and sum
to exactly 1.0, and unknown/missing fields are rejected. Color tables accept basic color names
or exact `#RRGGBB` keys. Season and style keys use the normalized item tags; category keys use
`top`, `bottom`, `outer`, `shoes`, and `accessory`.

The bundled default keeps the previous 40% color, 35% season, and 25% style weighting. Its
category weight is 0 until validated clothing correlation data is supplied. A missing color now
uses an explicit 60-point fallback instead of being treated as a high-scoring neutral color.

## Scope

This alpha release provides reusable contracts, deterministic local scoring, and optional
hosted LLM explanation adapters. It does not yet bundle a production segmentation or Vision
model. API keys must be supplied by the consuming application and must never be committed.
