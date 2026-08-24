from __future__ import annotations

from pathlib import Path

from PIL import Image

from cloth_vision_core.models import Category


class OutfitImageComposer:
    """Arrange garment images on a neutral canvas without regenerating them."""

    _cells = {
        Category.OUTER: (40, 40, 500, 820),
        Category.TOP: (400, 80, 820, 470),
        Category.BOTTOM: (730, 40, 1160, 750),
        Category.SHOES: (390, 570, 760, 850),
        Category.ACCESSORY: (810, 650, 1150, 860),
    }

    def compose(
        self,
        items: list[tuple[Category, Path]],
        destination: Path,
        *,
        size: tuple[int, int] = (1200, 900),
    ) -> Path:
        canvas = Image.new("RGB", size, "#F4F1EC")
        for index, (category, path) in enumerate(items):
            cell = self._cells.get(category, self._fallback_cell(index, size))
            with Image.open(path) as source:
                garment = source.convert("RGBA")
                garment.thumbnail((cell[2] - cell[0], cell[3] - cell[1]), Image.Resampling.LANCZOS)
                x = cell[0] + (cell[2] - cell[0] - garment.width) // 2
                y = cell[1] + (cell[3] - cell[1] - garment.height) // 2
                canvas.paste(garment, (x, y), garment)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="WEBP", quality=92, method=6)
        return destination

    @staticmethod
    def _fallback_cell(index: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = size
        column = index % 3
        row = (index // 3) % 2
        return (
            column * width // 3 + 30,
            row * height // 2 + 30,
            (column + 1) * width // 3 - 30,
            (row + 1) * height // 2 - 30,
        )
