from io import BytesIO

import pytest
from PIL import Image

from cloth_vision_core import InvalidImageError, PillowImageProcessor


def save_image(path, color: tuple[int, int, int], size: tuple[int, int] = (256, 256)) -> None:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())


def test_validates_image_without_inferring_a_background_color(tmp_path) -> None:
    path = tmp_path / "pink.png"
    save_image(path, (235, 100, 160))

    result = PillowImageProcessor().process(path)

    assert result.display_hex is None
    assert result.color_name is None
    assert result.width == 256


def test_rejects_small_image(tmp_path) -> None:
    path = tmp_path / "small.png"
    save_image(path, (0, 0, 0), (64, 64))

    with pytest.raises(InvalidImageError):
        PillowImageProcessor().process(path)
