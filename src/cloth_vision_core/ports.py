from __future__ import annotations

from pathlib import Path
from typing import Protocol

from cloth_vision_core.models import (
    ExplanationContext,
    ItemProfile,
    OutfitCandidate,
    OutfitExplanation,
    ProcessedImage,
    VisionResult,
)


class ImageProcessor(Protocol):
    def process(self, image_path: Path) -> ProcessedImage: ...


class VisionProvider(Protocol):
    def analyze(self, image: ProcessedImage) -> VisionResult: ...


class SegmentationProvider(Protocol):
    def segment(self, image: ProcessedImage) -> ProcessedImage: ...


class ExplanationProvider(Protocol):
    def explain(self, context: ExplanationContext) -> str: ...


class OutfitExplanationProvider(Protocol):
    def explain(
        self, candidate: OutfitCandidate, items: list[ItemProfile]
    ) -> OutfitExplanation: ...


class TextGenerationProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
