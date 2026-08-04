from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from cloth_vision_core.errors import InvalidImageError
from cloth_vision_core.models import ProcessedImage


def color_name(red: int, green: int, blue: int) -> str:
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    if value < 0.18:
        return "black"
    if saturation < 0.12:
        return "white" if value > 0.85 else "gray"
    if hue < 0.04 or hue >= 0.96:
        return "red"
    if hue < 0.10:
        return "orange"
    if hue < 0.18:
        return "yellow"
    if hue < 0.45:
        return "green"
    if hue < 0.72:
        return "blue"
    if hue < 0.88:
        return "purple"
    return "pink"


class PillowImageProcessor:
    def __init__(self, minimum_size: int = 128, analysis_size: int = 128) -> None:
        self.minimum_size = minimum_size
        self.analysis_size = analysis_size

    def process(self, image_path: Path) -> ProcessedImage:
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                if width < self.minimum_size or height < self.minimum_size:
                    raise InvalidImageError(
                        f"image resolution must be at least {self.minimum_size}x{self.minimum_size}"
                    )
                image.thumbnail((self.analysis_size, self.analysis_size))
                colors = image.quantize(colors=5).convert("RGB").getcolors()
                if not colors:
                    raise InvalidImageError("unable to extract image colors")
                _, (red, green, blue) = max(colors, key=lambda item: item[0])
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("image is damaged or unsupported") from exc

        return ProcessedImage(
            path=image_path,
            width=width,
            height=height,
            display_hex=f"#{red:02X}{green:02X}{blue:02X}",
            color_name=color_name(red, green, blue),
        )
