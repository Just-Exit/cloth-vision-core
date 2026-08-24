from uuid import uuid4

from cloth_vision_core import (
    Category,
    ItemProfile,
    MatchingEngine,
    OutfitRecommendationEngine,
)


def item(category: Category, color: str, style: str) -> ItemProfile:
    return ItemProfile(
        id=uuid4(),
        category=category,
        color_hex=color,
        style_tags=[style],
        season_tags=["spring"],
    )


def test_recommends_three_diverse_cross_category_outfits() -> None:
    tops = [
        item(Category.TOP, "#111111", "casual"),
        item(Category.TOP, "#FFFFFF", "minimal"),
        item(Category.TOP, "#223366", "classic"),
    ]
    bottoms = [
        item(Category.BOTTOM, "#224477", "casual"),
        item(Category.BOTTOM, "#222222", "minimal"),
    ]
    outer = item(Category.OUTER, "#888888", "classic")
    shoes = item(Category.SHOES, "#FFFFFF", "casual")

    result = OutfitRecommendationEngine(MatchingEngine()).recommend(
        [*tops, *bottoms, outer, shoes], limit=3
    )

    assert len(result.outfits) == 3
    assert len({outfit.item_ids[0] for outfit in result.outfits}) == 3
    assert all(len(outfit.item_ids) == 4 for outfit in result.outfits)
    assert result.missing_categories == []
    assert result.evaluated_candidates < 100


def test_reports_missing_required_category() -> None:
    result = OutfitRecommendationEngine(MatchingEngine()).recommend(
        [item(Category.TOP, "#111111", "casual")]
    )

    assert result.outfits == []
    assert result.missing_categories == [Category.BOTTOM]
    assert result.evaluated_candidates == 0
