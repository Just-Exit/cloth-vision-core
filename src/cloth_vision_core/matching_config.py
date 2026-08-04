from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cloth_vision_core.errors import InvalidMatchingConfigError

_COMPONENTS = {"color", "season", "style", "category"}


@dataclass(frozen=True, slots=True)
class _ColorRules:
    correlations: Mapping[str, Mapping[str, int]]
    missing_score: int
    neutral_score: int
    minimum_score: int


@dataclass(frozen=True, slots=True)
class _TagRules:
    correlations: Mapping[str, Mapping[str, int]]
    missing_score: int
    default_score: int
    same_score: int


@dataclass(frozen=True, slots=True)
class _StyleRules:
    correlations: Mapping[str, Mapping[str, int]]
    default_score: int
    base_match_score: int
    per_shared_tag: int
    maximum_score: int


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    """Validated, versioned scoring rules loaded from JSON-compatible data."""

    version: int
    overall_weights: Mapping[str, float]
    color: _ColorRules
    season: _TagRules
    style: _StyleRules
    category: _TagRules

    @classmethod
    def default(cls) -> MatchingConfig:
        resource = resources.files("cloth_vision_core.data").joinpath("default_matching.json")
        try:
            data = json.loads(resource.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidMatchingConfigError("unable to load the default matching config") from exc
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> MatchingConfig:
        try:
            with Path(path).open(encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidMatchingConfigError(f"unable to load matching config: {path}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatchingConfig:
        root = _mapping(data, "config")
        _check_keys(
            root,
            {"version", "overall_weights", "color", "season", "style", "category"},
            "config",
        )
        version = root["version"]
        if version != 1 or isinstance(version, bool):
            raise InvalidMatchingConfigError("config.version must be the integer 1")

        weights_data = _mapping(root["overall_weights"], "overall_weights")
        if set(weights_data) != _COMPONENTS:
            raise InvalidMatchingConfigError(
                "overall_weights must contain exactly: category, color, season, style"
            )
        weights = MappingProxyType(
            {
                name: _weight(value, f"overall_weights.{name}")
                for name, value in weights_data.items()
            }
        )
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise InvalidMatchingConfigError("overall_weights must sum to 1.0")

        color_data = _mapping(root["color"], "color")
        _check_keys(
            color_data,
            {"correlations", "missing_score", "neutral_score", "minimum_score"},
            "color",
        )
        color = _ColorRules(
            correlations=_correlations(color_data["correlations"], "color.correlations"),
            missing_score=_score(color_data["missing_score"], "color.missing_score"),
            neutral_score=_score(color_data["neutral_score"], "color.neutral_score"),
            minimum_score=_score(color_data["minimum_score"], "color.minimum_score"),
        )

        season = _tag_rules(root["season"], "season")
        category = _tag_rules(root["category"], "category")

        style_data = _mapping(root["style"], "style")
        _check_keys(
            style_data,
            {
                "correlations",
                "default_score",
                "base_match_score",
                "per_shared_tag",
                "maximum_score",
            },
            "style",
        )
        style = _StyleRules(
            correlations=_correlations(style_data["correlations"], "style.correlations"),
            default_score=_score(style_data["default_score"], "style.default_score"),
            base_match_score=_score(style_data["base_match_score"], "style.base_match_score"),
            per_shared_tag=_score(style_data["per_shared_tag"], "style.per_shared_tag"),
            maximum_score=_score(style_data["maximum_score"], "style.maximum_score"),
        )
        if style.base_match_score > style.maximum_score:
            raise InvalidMatchingConfigError(
                "style.base_match_score must not exceed style.maximum_score"
            )

        return cls(
            version=1,
            overall_weights=weights,
            color=color,
            season=season,
            style=style,
            category=category,
        )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise InvalidMatchingConfigError(f"{field} must be a JSON object")
    return value


def _check_keys(data: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing:
        raise InvalidMatchingConfigError(f"{field} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise InvalidMatchingConfigError(
            f"{field} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _score(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise InvalidMatchingConfigError(f"{field} must be an integer between 0 and 100")
    return value


def _weight(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise InvalidMatchingConfigError(f"{field} must be a number between 0 and 1")
    return float(value)


def _correlations(value: Any, field: str) -> Mapping[str, Mapping[str, int]]:
    data = _mapping(value, field)
    normalized: dict[str, dict[str, int]] = {}
    for left, raw_targets in data.items():
        left_key = _key(left, field)
        if left_key in normalized:
            raise InvalidMatchingConfigError(f"{field} has a duplicate normalized key: {left}")
        targets = _mapping(raw_targets, f"{field}.{left}")
        normalized_targets: dict[str, int] = {}
        for right, score in targets.items():
            right_key = _key(right, field)
            if right_key in normalized_targets:
                raise InvalidMatchingConfigError(
                    f"{field}.{left} has a duplicate normalized key: {right}"
                )
            normalized_targets[right_key] = _score(score, f"{field}.{left}.{right}")
        normalized[left_key] = normalized_targets
    return MappingProxyType(
        {left: MappingProxyType(targets) for left, targets in normalized.items()}
    )


def _key(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidMatchingConfigError(f"{field} keys must not be empty")
    return normalized.upper() if normalized.startswith("#") else normalized.casefold()


def _tag_rules(value: Any, field: str) -> _TagRules:
    data = _mapping(value, field)
    _check_keys(
        data,
        {"correlations", "missing_score", "default_score", "same_score"},
        field,
    )
    return _TagRules(
        correlations=_correlations(data["correlations"], f"{field}.correlations"),
        missing_score=_score(data["missing_score"], f"{field}.missing_score"),
        default_score=_score(data["default_score"], f"{field}.default_score"),
        same_score=_score(data["same_score"], f"{field}.same_score"),
    )
