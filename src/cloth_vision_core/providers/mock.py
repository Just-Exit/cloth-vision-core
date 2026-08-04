from __future__ import annotations

from cloth_vision_core.models import Category, ProcessedImage, VisionResult


class MockVisionProvider:
    """Deterministic provider for examples, tests, and offline demos."""

    def __init__(self, result: VisionResult | None = None) -> None:
        self.result = result or VisionResult(
            category=Category.OUTER,
            subcategory="puffer_jacket",
            style_tags=["casual"],
            season_tags=["winter"],
            confidence=0.9,
        )

    def analyze(self, image: ProcessedImage) -> VisionResult:
        del image
        return self.result
