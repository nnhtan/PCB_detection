from pathlib import Path
from typing import Any

from backend.app.core.config import MODEL_PATH


class Detector:
    """Lazy wrapper around the configured Ultralytics YOLO model."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required to load the YOLO detector."
            ) from exc

        self._model: Any = YOLO(str(model_path))

    def predict(self, image: Any) -> Any:
        """Run inference on an image and return the model's raw results."""
        return self._model.predict(image)
