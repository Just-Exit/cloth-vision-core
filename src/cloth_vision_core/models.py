from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class Category(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    OUTER = "outer"
    SHOES = "shoes"
    ACCESSORY = "accessory"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    path: Path
    width: int
    height: int
    display_hex: str
    color_name: str
    analysis_path: Path | None = None
    mask_path: Path | None = None
    transparent_path: Path | None = None
    bounding_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class ItemColor:
    display_hex: str
    color_name: str
    ratio: float
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class MaterialEstimate:
    name: str
    confidence: float | None = None
    source: str = "vision_estimate"


@dataclass(frozen=True, slots=True)
class VisionResult:
    category: Category
    subcategory: str
    style_tags: list[str] = field(default_factory=list)
    season_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    attributes: dict[str, str] = field(default_factory=dict)
    colors: list[ItemColor] = field(default_factory=list)
    materials: list[MaterialEstimate] = field(default_factory=list)
    suggested_display_name: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    category: Category
    subcategory: str
    color_hex: str
    color_name: str
    style_tags: list[str]
    season_tags: list[str]
    confidence: float
    attributes: dict[str, str] = field(default_factory=dict)
    colors: list[ItemColor] = field(default_factory=list)
    materials: list[MaterialEstimate] = field(default_factory=list)
    suggested_display_name: str | None = None


@dataclass(frozen=True, slots=True)
class ItemProfile:
    id: UUID
    category: Category = Category.UNKNOWN
    color_hex: str | None = None
    style_tags: list[str] = field(default_factory=list)
    season_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MatchResult:
    source_item_id: UUID
    target_item_id: UUID
    overall_score: int
    color_score: int
    season_score: int
    style_score: int
    reasons: list[str]
    category_score: int = 60


@dataclass(frozen=True, slots=True)
class ExplanationContext:
    match: MatchResult
    source: ItemProfile
    target: ItemProfile
