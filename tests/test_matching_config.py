import json
from uuid import uuid4

import pytest

from cloth_vision_core import (
    Category,
    InvalidMatchingConfigError,
    ItemProfile,
    MatchingConfig,
    MatchingEngine,
)


def custom_config() -> dict:
    return {
        "version": 1,
        "overall_weights": {
            "color": 0.25,
            "season": 0.25,
            "style": 0.25,
            "category": 0.25,
        },
        "color": {
            "correlations": {"red": {"blue": 90}},
            "missing_score": 50,
            "neutral_score": 92,
            "minimum_score": 55,
        },
        "season": {
            "correlations": {"winter": {"spring": 80}},
            "missing_score": 50,
            "default_score": 60,
            "same_score": 100,
        },
        "style": {
            "correlations": {"casual": {"formal": 75}},
            "default_score": 60,
            "base_match_score": 70,
            "per_shared_tag": 15,
            "maximum_score": 100,
        },
        "category": {
            "correlations": {"top": {"bottom": 95}},
            "missing_score": 50,
            "default_score": 60,
            "same_score": 100,
        },
    }


def test_json_correlations_drive_every_matching_component(tmp_path) -> None:
    path = tmp_path / "matching.json"
    path.write_text(json.dumps(custom_config()), encoding="utf-8")
    source = ItemProfile(
        id=uuid4(),
        category=Category.TOP,
        color_hex="#FF0000",
        season_tags=["winter"],
        style_tags=["casual"],
    )
    target = ItemProfile(
        id=uuid4(),
        category=Category.BOTTOM,
        color_hex="#0000FF",
        season_tags=["spring"],
        style_tags=["formal"],
    )

    result = MatchingEngine.from_json(path).compare(source, target)

    assert result.color_score == 90
    assert result.season_score == 80
    assert result.style_score == 75
    assert result.category_score == 95
    assert result.overall_score == 85
    assert len(result.reasons) == 4


def test_correlations_are_symmetric() -> None:
    config = MatchingConfig.from_dict(custom_config())
    source = ItemProfile(id=uuid4(), color_hex="#0000FF")
    target = ItemProfile(id=uuid4(), color_hex="#FF0000")

    result = MatchingEngine(config).compare(source, target)

    assert result.color_score == 90


def test_missing_color_uses_explicit_missing_score() -> None:
    source = ItemProfile(id=uuid4(), color_hex=None)
    target = ItemProfile(id=uuid4(), color_hex="#FFFFFF")

    result = MatchingEngine().compare(source, target)

    assert result.color_score == 60


def test_invalid_weight_sum_is_rejected() -> None:
    data = custom_config()
    data["overall_weights"]["category"] = 0.5

    with pytest.raises(InvalidMatchingConfigError, match="sum to 1.0"):
        MatchingConfig.from_dict(data)


def test_invalid_hex_is_rejected() -> None:
    source = ItemProfile(id=uuid4(), color_hex="#INVALID")
    target = ItemProfile(id=uuid4(), color_hex="#FFFFFF")

    with pytest.raises(ValueError, match="#RRGGBB"):
        MatchingEngine().compare(source, target)
