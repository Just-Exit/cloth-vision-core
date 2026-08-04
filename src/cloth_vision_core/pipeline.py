from __future__ import annotations

from pathlib import Path

from cloth_vision_core.models import AnalysisResult, Category, VisionResult
from cloth_vision_core.ports import ImageProcessor, VisionProvider


class AnalysisPipeline:
    def __init__(
        self,
        image_processor: ImageProcessor,
        vision_provider: VisionProvider | None = None,
    ) -> None:
        self.image_processor = image_processor
        self.vision_provider = vision_provider

    def analyze(self, image_path: Path) -> AnalysisResult:
        image = self.image_processor.process(image_path)
        vision = (
            self.vision_provider.analyze(image)
            if self.vision_provider
            else VisionResult(
                category=Category.UNKNOWN,
                subcategory="unclassified",
                confidence=0.0,
            )
        )
        return AnalysisResult(
            category=vision.category,
            subcategory=vision.subcategory,
            color_hex=image.display_hex,
            color_name=image.color_name,
            style_tags=vision.style_tags,
            season_tags=vision.season_tags,
            confidence=vision.confidence,
            attributes=vision.attributes,
        )
