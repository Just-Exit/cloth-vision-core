from __future__ import annotations

import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw

from cloth_vision_core import PillowImageProcessor, RembgSegmentationProvider


def test_segmentation_preserves_original_and_creates_analysis_artifacts(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "garment.png"
    Image.new("RGB", (256, 256), "white").save(path)

    def remove(image, session=None):
        del session
        result = image.convert("RGBA")
        alpha = Image.new("L", image.size, 0)
        ImageDraw.Draw(alpha).rectangle((64, 32, 191, 223), fill=255)
        result.putalpha(alpha)
        return result

    monkeypatch.setitem(
        sys.modules,
        "rembg",
        SimpleNamespace(remove=remove, new_session=lambda model: model),
    )
    before = path.read_bytes()
    processed = RembgSegmentationProvider().segment(PillowImageProcessor().process(path))

    assert path.read_bytes() == before
    assert processed.bounding_box == (64, 32, 192, 224)
    assert processed.mask_path and processed.mask_path.exists()
    assert processed.transparent_path and processed.transparent_path.exists()
    assert processed.analysis_path and processed.analysis_path.exists()
    assert processed.thumbnail_path and processed.thumbnail_path.exists()
    with Image.open(processed.analysis_path) as normalized:
        assert normalized.size == (1024, 1024)
    with Image.open(processed.thumbnail_path) as thumbnail:
        assert thumbnail.size == (384, 384)
