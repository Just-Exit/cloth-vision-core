from pathlib import Path

from cloth_vision_core import AnalysisPipeline, PillowImageProcessor
from cloth_vision_core.providers import MockVisionProvider

pipeline = AnalysisPipeline(
    image_processor=PillowImageProcessor(),
    vision_provider=MockVisionProvider(),
)
result = pipeline.analyze(Path("item.jpg"))
print(result)
