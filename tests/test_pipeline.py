from io import BytesIO

from PIL import Image

from cloth_vision_core import AnalysisPipeline, Category, ItemColor, PillowImageProcessor
from cloth_vision_core.errors import ProviderError
from cloth_vision_core.providers import MockVisionProvider


def test_pipeline_combines_local_image_and_provider_results(tmp_path) -> None:
    path = tmp_path / "item.png"
    buffer = BytesIO()
    Image.new("RGB", (256, 256), (20, 40, 80)).save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())

    result = AnalysisPipeline(
        PillowImageProcessor(),
        MockVisionProvider(),
    ).analyze(path)

    assert result.category == Category.OUTER
    assert result.subcategory == "puffer_jacket"
    assert result.season_tags == ["winter"]
    assert result.color_hex is None


def test_pipeline_marks_analysis_unavailable_when_provider_fails(tmp_path) -> None:
    path = tmp_path / "garment.png"
    Image.new("RGB", (256, 256), (30, 80, 180)).save(path)

    class FailingProvider:
        def analyze(self, image):
            del image
            raise ProviderError("temporary failure")

    result = AnalysisPipeline(PillowImageProcessor(), FailingProvider()).analyze(path)

    assert result.category is Category.UNKNOWN
    assert result.color_hex is None
    assert result.color_name is None
    assert result.colors == []
    assert result.attributes == {"analysis_warning": "vision_provider_failed"}


def test_pipeline_uses_largest_vision_palette_color_as_primary(tmp_path) -> None:
    path = tmp_path / "garment.png"
    Image.new("RGB", (256, 256), "white").save(path)

    class PaletteProvider:
        def analyze(self, image):
            del image
            from cloth_vision_core import VisionResult

            return VisionResult(
                category=Category.TOP,
                subcategory="t-shirt",
                colors=[
                    ItemColor("#FFFFFF", "white", 0.08, 0.92),
                    ItemColor("#1B1B2A", "dark navy", 0.88, 0.96),
                ],
            )

    result = AnalysisPipeline(PillowImageProcessor(), PaletteProvider()).analyze(path)

    assert result.color_hex == "#1B1B2A"
    assert result.color_name == "dark navy"
