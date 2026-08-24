from __future__ import annotations

from itertools import combinations, product
from statistics import mean
from uuid import UUID

from cloth_vision_core.matching import MatchingEngine
from cloth_vision_core.models import (
    Category,
    ItemProfile,
    MatchResult,
    OutfitCandidate,
    OutfitRecommendationResult,
)


class OutfitRecommendationEngine:
    """Build a small, diverse set of outfits without enumerating the full Cartesian product."""

    def __init__(self, matching_engine: MatchingEngine, *, beam_width: int = 20) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be greater than zero")
        self.matching_engine = matching_engine
        self.beam_width = beam_width

    def recommend(
        self,
        items: list[ItemProfile],
        *,
        limit: int = 3,
    ) -> OutfitRecommendationResult:
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        by_category = {
            category: [item for item in items if item.category is category] for category in Category
        }
        missing = [
            category for category in (Category.TOP, Category.BOTTOM) if not by_category[category]
        ]
        if missing:
            return OutfitRecommendationResult([], missing, 0)

        pair_cache: dict[tuple[str, str], MatchResult] = {}
        evaluated = 0
        candidates = []
        for top, bottom in product(by_category[Category.TOP], by_category[Category.BOTTOM]):
            candidates.append(self._score((top, bottom), pair_cache))
            evaluated += 1
        candidates = self._top(candidates)

        for category in (Category.OUTER, Category.SHOES, Category.ACCESSORY):
            optional_items = by_category[category]
            if not optional_items:
                continue
            profiles_by_id = {item.id: item for item in items}
            expanded = []
            for candidate, optional in product(candidates, optional_items):
                profiles = tuple(profiles_by_id[item_id] for item_id in candidate.item_ids)
                expanded.append(self._score((*profiles, optional), pair_cache))
                evaluated += 1
            candidates = self._top(expanded)

        return OutfitRecommendationResult(
            outfits=self._diverse(candidates, limit),
            missing_categories=[],
            evaluated_candidates=evaluated,
        )

    def _score(
        self,
        items: tuple[ItemProfile, ...],
        cache: dict[tuple[str, str], MatchResult],
    ) -> OutfitCandidate:
        matches = [self._pair(left, right, cache) for left, right in combinations(items, 2)]
        reasons = list(dict.fromkeys(reason for match in matches for reason in match.reasons))[:4]
        return OutfitCandidate(
            item_ids=tuple(item.id for item in items),
            overall_score=round(mean(match.overall_score for match in matches)),
            color_score=round(mean(match.color_score for match in matches)),
            season_score=round(mean(match.season_score for match in matches)),
            style_score=round(mean(match.style_score for match in matches)),
            reasons=reasons,
        )

    def _pair(
        self,
        left: ItemProfile,
        right: ItemProfile,
        cache: dict[tuple[str, str], MatchResult],
    ) -> MatchResult:
        key = tuple(sorted((str(left.id), str(right.id))))
        if key not in cache:
            cache[key] = self.matching_engine.compare(left, right)
        return cache[key]

    def _top(self, candidates: list[OutfitCandidate]) -> list[OutfitCandidate]:
        return sorted(candidates, key=lambda item: item.overall_score, reverse=True)[
            : self.beam_width
        ]

    @staticmethod
    def _diverse(candidates: list[OutfitCandidate], limit: int) -> list[OutfitCandidate]:
        selected = []
        used_tops: set[UUID] = set()
        for candidate in candidates:
            top_id = candidate.item_ids[0]
            if top_id not in used_tops:
                selected.append(candidate)
                used_tops.add(top_id)
            if len(selected) == limit:
                return selected
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == limit:
                break
        return selected
