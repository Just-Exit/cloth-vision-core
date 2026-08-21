from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from cloth_vision_core.errors import ProviderError
from cloth_vision_core.models import (
    Category,
    ItemColor,
    MaterialEstimate,
    ProcessedImage,
    VisionResult,
)

SYSTEM_PROMPT = """You are a garment analysis system for a digital wardrobe.
Analyze the primary garment, not the person or background. Infer only visually supported facts.
Use null-like omissions or empty arrays when uncertain. Never invent brand, price, or exact fabric
composition. Color values must describe the garment. Enum values must use the supplied English
values. Keep tags short, lowercase, and stable."""


VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "display_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
            "description": "Concise natural Korean garment name",
        },
        "category": {"type": "string", "enum": [item.value for item in Category]},
        "subcategory": {"type": "string", "maxLength": 60},
        "colors": {
            "type": "array",
            "maxItems": 5,
            "description": "Garment colors ordered by descending visible area ratio",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "display_hex": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                    "color_name": {"type": "string", "maxLength": 50},
                    "ratio": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["display_hex", "color_name", "ratio", "confidence"],
            },
        },
        "materials": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "maxLength": 50},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["name", "confidence"],
            },
        },
        "style_tags": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "season_tags": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "enum": ["spring", "summer", "fall", "winter"]},
        },
        "attributes": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pattern": {"type": "string"},
                "fit": {"type": "string"},
                "sleeve": {"type": "string"},
                "length": {"type": "string"},
            },
            "required": ["pattern", "fit", "sleeve", "length"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "display_name", "category", "subcategory", "colors", "materials", "style_tags",
        "season_tags", "attributes", "confidence",
    ],
}


def _data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class OpenAIVisionProvider:
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
                    "OpenAI vision support requires the 'openai' extra"
                ) from exc
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self._client = client

    def analyze(self, image: ProcessedImage) -> VisionResult:
        paths = [image.path]
        analysis_path = image.analysis_path
        if analysis_path and analysis_path != image.path:
            paths.append(analysis_path)
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Analyze the garment. The first image is the original; a second image, "
                    "when present, is an isolated crop."
                ),
            }
        ]
        content.extend(
            {"type": "input_image", "image_url": _data_url(path), "detail": "high"}
            for path in paths
        )
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "garment_analysis",
                        "strict": True,
                        "schema": VISION_SCHEMA,
                    }
                },
            )
            payload = json.loads(response.output_text)
            return self._result(payload)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI garment analysis failed") from exc

    @staticmethod
    def _result(payload: dict[str, Any]) -> VisionResult:
        try:
            return VisionResult(
                category=Category(payload["category"]),
                subcategory=str(payload["subcategory"]),
                suggested_display_name=str(payload["display_name"]),
                colors=[
                    ItemColor(
                        display_hex=item["display_hex"].upper(),
                        color_name=item["color_name"],
                        ratio=float(item["ratio"]),
                        confidence=float(item["confidence"]),
                    )
                    for item in payload["colors"]
                ],
                materials=[
                    MaterialEstimate(
                        name=item["name"], confidence=float(item["confidence"])
                    )
                    for item in payload["materials"]
                ],
                style_tags=list(payload["style_tags"]),
                season_tags=list(payload["season_tags"]),
                attributes={str(key): str(value) for key, value in payload["attributes"].items()},
                confidence=float(payload["confidence"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("OpenAI returned an invalid garment analysis") from exc
