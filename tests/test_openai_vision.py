from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cloth_vision_core import ProcessedImage
from cloth_vision_core.providers import OpenAIVisionProvider


def test_openai_vision_sends_image_and_parses_structured_result(tmp_path: Path) -> None:
    image_path = tmp_path / "shirt.jpg"
    image_path.write_bytes(b"image")
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "display_name": "네이비 코튼 셔츠",
                        "category": "top",
                        "subcategory": "shirt",
                        "colors": [
                            {
                                "display_hex": "#123456",
                                "color_name": "navy",
                                "ratio": 0.9,
                                "confidence": 0.8,
                            }
                        ],
                        "materials": [{"name": "cotton", "confidence": 0.7}],
                        "style_tags": ["casual"],
                        "season_tags": ["spring"],
                        "attributes": {
                            "pattern": "solid",
                            "fit": "regular",
                            "sleeve": "long",
                            "length": "regular",
                        },
                        "confidence": 0.86,
                    }
                )
            )

    provider = OpenAIVisionProvider(
        model="test-model", client=SimpleNamespace(responses=Responses())
    )
    result = provider.analyze(
        ProcessedImage(
            path=image_path,
            width=200,
            height=300,
            display_hex="#123456",
            color_name="blue",
        )
    )

    assert result.category.value == "top"
    assert result.suggested_display_name == "네이비 코튼 셔츠"
    assert result.materials[0].name == "cotton"
    assert result.colors[0].display_hex == "#123456"
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["input"][0]["content"][1]["type"] == "input_image"
