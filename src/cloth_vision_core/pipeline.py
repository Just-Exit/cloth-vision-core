from __future__ import annotations

from pathlib import Path

from cloth_vision_core.errors import ProviderError
from cloth_vision_core.models import AnalysisResult, Category, ItemColor, VisionResult
from cloth_vision_core.ports import ImageProcessor, SegmentationProvider, VisionProvider


class AnalysisPipeline:
    def __init__(
        self,
        image_processor: ImageProcessor,
        vision_provider: VisionProvider | None = None,
        segmentation_provider: SegmentationProvider | None = None,
    ) -> None:
        self.image_processor = image_processor
        self.vision_provider = vision_provider
        self.segmentation_provider = segmentation_provider

    def analyze(self, image_path: Path) -> AnalysisResult:
        image = self.image_processor.process(image_path)
        if self.segmentation_provider:
            try:
                image = self.segmentation_provider.segment(image)
            except ProviderError:
                # Segmentation improves analysis, but the original image remains a valid fallback.
                pass
        try:
            vision = (
                self.vision_provider.analyze(image)
                if self.vision_provider
                else self._fallback_vision("vision_provider_unavailable")
            )
        except ProviderError:
            vision = self._fallback_vision("vision_provider_failed")
        primary_color = max(vision.colors, key=lambda color: color.ratio) if vision.colors else None
        return AnalysisResult(
            category=vision.category,
            subcategory=vision.subcategory,
            color_hex=primary_color.display_hex if primary_color else image.display_hex,
            color_name=primary_color.color_name if primary_color else image.color_name,
            style_tags=vision.style_tags,
            season_tags=vision.season_tags,
            confidence=vision.confidence,
            attributes=vision.attributes,
            colors=vision.colors or self._segmented_color(image),
            materials=vision.materials,
            suggested_display_name=vision.suggested_display_name,
        )

    @staticmethod
    def _segmented_color(image) -> list[ItemColor]:
        if image.mask_path and image.display_hex and image.color_name:
            return [
                ItemColor(
                    display_hex=image.display_hex,
                    color_name=image.color_name,
                    ratio=1.0,
                    confidence=1.0,
                )
            ]
        return []

    @staticmethod
    def _fallback_vision(warning: str | None = None) -> VisionResult:
        return VisionResult(
            category=Category.UNKNOWN,
            subcategory="unclassified",
            confidence=0.0,
            attributes={"analysis_warning": warning} if warning else {},
        )
