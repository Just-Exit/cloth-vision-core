from io import BytesIO

from PIL import Image

from cloth_vision_core import AnalysisPipeline, Category, PillowImageProcessor
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
    assert result.color_hex == "#142850"
