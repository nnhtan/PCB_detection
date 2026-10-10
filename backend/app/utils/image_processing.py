from typing import Any

import cv2
import numpy as np


def decode_image(contents: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR image."""
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded content is not a valid image.")
    return image


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw xyxy boxes and labels on a copy of an image."""
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["box"])
        label = str(detection["label"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return output
