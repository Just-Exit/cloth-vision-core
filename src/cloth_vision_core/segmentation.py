from __future__ import annotations

from collections import Counter
from typing import Any

from PIL import Image

from cloth_vision_core.errors import ProviderError
from cloth_vision_core.image import color_name
from cloth_vision_core.models import ProcessedImage


class RembgSegmentationProvider:
    """Create non-generative garment artifacts while preserving original RGB pixels."""

    def __init__(
        self,
        *,
        model: str = "u2netp",
        session: Any | None = None,
        mask_threshold: int = 224,
    ) -> None:
        self.model = model
        self._session = session
        self.mask_threshold = mask_threshold

    def segment(self, image: ProcessedImage) -> ProcessedImage:
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise ProviderError("segmentation requires the 'segmentation' extra") from exc

        try:
            if self._session is None:
                self._session = new_session(self.model)
            with Image.open(image.path) as source:
                original = source.convert("RGB")
                isolated = remove(original, session=self._session).convert("RGBA")
                mask = isolated.getchannel("A")
                bbox = mask.point(
                    lambda value: 255 if value >= self.mask_threshold else 0
                ).getbbox()
                if not bbox:
                    raise ProviderError("segmentation did not find a garment")

                artifact_dir = image.path.parent / "derived"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                mask_path = artifact_dir / "mask.png"
                transparent_path = artifact_dir / "transparent.png"
                analysis_path = artifact_dir / "normalized.jpg"
                thumbnail_path = artifact_dir / "thumbnail.webp"
                mask.save(mask_path, format="PNG")
                isolated.save(transparent_path, format="PNG")

                crop = original.crop(bbox)
                crop_mask = mask.crop(bbox)
                canvas_size = 1024
                foreground_size = 768
                scale = min(foreground_size / crop.width, foreground_size / crop.height)
                resized_size = (
                    max(1, round(crop.width * scale)),
                    max(1, round(crop.height * scale)),
                )
                crop = crop.resize(resized_size, Image.Resampling.LANCZOS)
                crop_mask = crop_mask.resize(resized_size, Image.Resampling.LANCZOS)
                normalized = Image.new("RGB", (canvas_size, canvas_size), (247, 247, 245))
                offset = (
                    (canvas_size - resized_size[0]) // 2,
                    (canvas_size - resized_size[1]) // 2,
                )
                normalized.paste(crop, box=offset, mask=crop_mask)
                normalized.save(analysis_path, format="JPEG", quality=92)
                thumbnail = normalized.resize((384, 384), Image.Resampling.LANCZOS)
                thumbnail.save(thumbnail_path, format="WEBP", quality=85, method=6)

                display_hex, display_name = self._dominant_color(original, mask)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("garment segmentation failed") from exc

        return ProcessedImage(
            path=image.path,
            width=image.width,
            height=image.height,
            display_hex=display_hex,
            color_name=display_name,
            analysis_path=analysis_path,
            mask_path=mask_path,
            transparent_path=transparent_path,
            thumbnail_path=thumbnail_path,
            bounding_box=bbox,
        )

    def _dominant_color(self, original: Image.Image, mask: Image.Image) -> tuple[str, str]:
        sample = original.copy()
        sample.thumbnail((192, 192))
        sample_mask = mask.copy()
        sample_mask.thumbnail(sample.size)
        pixels = [
            rgb
            for rgb, alpha in zip(
                sample.get_flattened_data(), sample_mask.get_flattened_data(), strict=True
            )
            if alpha >= self.mask_threshold
        ]
        if not pixels:
            raise ProviderError("segmentation mask contains no usable pixels")
        # Coarse bins reduce camera noise without changing the stored source pixels.
        bins = Counter(tuple((channel // 16) * 16 + 8 for channel in rgb) for rgb in pixels)
        red, green, blue = bins.most_common(1)[0][0]
        return f"#{red:02X}{green:02X}{blue:02X}", color_name(red, green, blue)
