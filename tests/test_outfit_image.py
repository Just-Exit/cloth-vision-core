from pathlib import Path

from PIL import Image

from cloth_vision_core import Category, OutfitImageComposer


def _garment(path: Path, color: str, size: tuple[int, int]) -> Path:
    Image.new("RGBA", size, color).save(path)
    return path


def test_composes_outfit_as_webp(tmp_path: Path) -> None:
    top = _garment(tmp_path / "top.png", "#CC0000", (200, 160))
    bottom = _garment(tmp_path / "bottom.png", "#0011CC", (140, 300))

    destination = OutfitImageComposer().compose(
        [(Category.TOP, top), (Category.BOTTOM, bottom)], tmp_path / "outfit.webp"
    )

    assert destination.is_file()
    with Image.open(destination) as image:
        assert image.size == (1200, 900)
        assert image.mode == "RGB"
        colors = image.resize((60, 45)).getcolors(maxcolors=3000)
        assert colors is not None and len(colors) > 2
