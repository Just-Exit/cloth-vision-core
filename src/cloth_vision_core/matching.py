from __future__ import annotations

import colorsys
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from cloth_vision_core.image import color_name
from cloth_vision_core.matching_config import MatchingConfig
from cloth_vision_core.models import Category, ItemProfile, MatchResult


class _TagScoreRules(Protocol):
    correlations: Mapping[str, Mapping[str, int]]
    missing_score: int
    default_score: int
    same_score: int


def _normalize_hex(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if len(normalized) != 7 or not normalized.startswith("#"):
        raise ValueError("color_hex must use the #RRGGBB format")
    try:
        int(normalized[1:], 16)
    except ValueError as exc:
        raise ValueError("color_hex must use the #RRGGBB format") from exc
    return normalized


def _hex_to_hsv(value: str) -> tuple[float, float, float]:
    red, green, blue = (int(value[index : index + 2], 16) / 255 for index in (1, 3, 5))
    return colorsys.rgb_to_hsv(red, green, blue)


def _color_keys(value: str) -> tuple[str, str]:
    red, green, blue = (int(value[index : index + 2], 16) for index in (1, 3, 5))
    return value, color_name(red, green, blue)


def _lookup(
    correlations: Mapping[str, Mapping[str, int]],
    left: str,
    right: str,
) -> int | None:
    direct = correlations.get(left, {}).get(right)
    return direct if direct is not None else correlations.get(right, {}).get(left)


def _color_score(left: str | None, right: str | None, config: MatchingConfig) -> int:
    left_hex = _normalize_hex(left)
    right_hex = _normalize_hex(right)
    if left_hex is None or right_hex is None:
        return config.color.missing_score

    left_hex_key, left_name = _color_keys(left_hex)
    right_hex_key, right_name = _color_keys(right_hex)
    configured = _lookup(config.color.correlations, left_hex_key, right_hex_key)
    if configured is None:
        configured = _lookup(config.color.correlations, left_name, right_name)
    if configured is not None:
        return configured

    left_hue, left_saturation, _ = _hex_to_hsv(left_hex)
    right_hue, right_saturation, _ = _hex_to_hsv(right_hex)
    if left_saturation < 0.18 or right_saturation < 0.18:
        return config.color.neutral_score
    distance = min(abs(left_hue - right_hue), 1 - abs(left_hue - right_hue))
    return max(config.color.minimum_score, round(100 - distance * 100))


def _normalize_tags(values: list[str]) -> set[str]:
    return {value.strip().casefold() for value in values if value.strip()}


def _tag_score(left: set[str], right: set[str], rules: _TagScoreRules) -> int:
    missing_score = rules.missing_score
    if not left or not right:
        return missing_score
    scores = []
    for left_tag in left:
        for right_tag in right:
            configured = _lookup(rules.correlations, left_tag, right_tag)
            if configured is not None:
                scores.append(configured)
            elif left_tag == right_tag:
                scores.append(rules.same_score)
            else:
                scores.append(rules.default_score)
    return max(scores)


def _style_score(left: set[str], right: set[str], config: MatchingConfig) -> int:
    shared = left & right
    if shared:
        return min(
            config.style.maximum_score,
            config.style.base_match_score + config.style.per_shared_tag * len(shared),
        )
    configured = [
        score
        for left_tag in left
        for right_tag in right
        if (score := _lookup(config.style.correlations, left_tag, right_tag)) is not None
    ]
    return max(configured, default=config.style.default_score)


class MatchingEngine:
    def __init__(self, config: MatchingConfig | None = None) -> None:
        self.config = config or MatchingConfig.default()

    @classmethod
    def from_json(cls, path: str | Path) -> MatchingEngine:
        return cls(MatchingConfig.from_json(path))

    def compare(self, source: ItemProfile, target: ItemProfile) -> MatchResult:
        source_seasons = _normalize_tags(source.season_tags)
        target_seasons = _normalize_tags(target.season_tags)
        source_styles = _normalize_tags(source.style_tags)
        target_styles = _normalize_tags(target.style_tags)

        color = _color_score(source.color_hex, target.color_hex, self.config)
        season = _tag_score(source_seasons, target_seasons, self.config.season)
        style = _style_score(source_styles, target_styles, self.config)
        source_category = None if source.category is Category.UNKNOWN else source.category.value
        target_category = None if target.category is Category.UNKNOWN else target.category.value
        category = _tag_score(
            {source_category} if source_category else set(),
            {target_category} if target_category else set(),
            self.config.category,
        )

        scores = {
            "color": color,
            "season": season,
            "style": style,
            "category": category,
        }
        overall = round(
            sum(scores[name] * weight for name, weight in self.config.overall_weights.items())
        )

        seasons = source_seasons & target_seasons
        styles = source_styles & target_styles
        reasons = []
        if seasons:
            reasons.append(f"{', '.join(sorted(seasons))} 계절에 함께 활용하기 좋습니다.")
        elif season > self.config.season.default_score:
            reasons.append("계절 조합의 상관관계 점수가 높습니다.")
        if styles:
            reasons.append(f"{', '.join(sorted(styles))} 스타일이 자연스럽게 연결됩니다.")
        elif style > self.config.style.default_score:
            reasons.append("스타일 조합의 상관관계 점수가 높습니다.")
        if color >= 85:
            reasons.append("색상 조합이 안정적입니다.")
        if self.config.overall_weights["category"] > 0 and category >= 85:
            reasons.append("의류 카테고리 조합이 잘 어울립니다.")
        if not reasons:
            reasons.append("기본 조합으로 활용할 수 있습니다.")

        return MatchResult(
            source_item_id=source.id,
            target_item_id=target.id,
            overall_score=overall,
            color_score=color,
            season_score=season,
            style_score=style,
            category_score=category,
            reasons=reasons,
        )
