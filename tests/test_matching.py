from uuid import uuid4

from cloth_vision_core import ItemProfile, MatchingEngine


def test_matching_returns_explainable_breakdown() -> None:
    source = ItemProfile(
        id=uuid4(),
        color_hex="#222222",
        season_tags=["winter"],
        style_tags=["casual"],
    )
    target = ItemProfile(
        id=uuid4(),
        color_hex="#FFFFFF",
        season_tags=["winter"],
        style_tags=["casual"],
    )

    result = MatchingEngine().compare(source, target)

    assert result.overall_score >= 85
    assert result.season_score == 100
    assert result.style_score == 85
    assert len(result.reasons) == 3
