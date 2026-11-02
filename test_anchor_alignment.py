import unittest

import cv2
import numpy as np

from backend.app.services.pcb_engine import (
    crop_board_to_reference_bounds,
    estimate_anchor_orientation,
    rotate_board_to_reference,
)


REFERENCE_COMPONENTS = [
    {"class": "capacitor", "x": 10.0, "y": 50.0},
    {"class": "capacitor", "x": 90.0, "y": 50.0},
    {"class": "IC", "x": 35.0, "y": 35.0},
]


def rotate_point(point, width, height, rotation):
    x, y = point
    if rotation == 0:
        return x, y
    if rotation == 90:
        return height - 1 - y, x
    if rotation == 180:
        return width - 1 - x, height - 1 - y
    return y, width - 1 - x


class AnchorAlignmentTests(unittest.TestCase):
    def test_selects_reference_orientation_for_quarter_turns(self):
        anchors = {
            "left": (10.0, 50.0),
            "right": (90.0, 50.0),
            "ic": (35.0, 35.0),
            "distractor": (60.0, 80.0),
        }
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation):
                detections = []
                for name, point in anchors.items():
                    x, y = rotate_point(point, 120, 100, rotation)
                    detections.append({
                        "class": "ic" if name == "ic" else "capacitor",
                        "cx": x,
                        "cy": y,
                        "confidence": 0.9,
                    })

                orientation, required, error = estimate_anchor_orientation(
                    REFERENCE_COMPONENTS, detections, 0.25
                )

                self.assertTrue(required)
                self.assertIsNone(error)
                self.assertIsNotNone(orientation)
                self.assertAlmostEqual(
                    orientation["left_cap"]["cx"],
                    rotate_point(anchors["left"], 120, 100, rotation)[0],
                )
                self.assertAlmostEqual(
                    orientation["ic"]["cx"],
                    rotate_point(anchors["ic"], 120, 100, rotation)[0],
                )

                image = np.zeros((100, 120, 3), dtype=np.uint8)
                if rotation == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif rotation == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated, transform = rotate_board_to_reference(
                    image, orientation["rotation_degrees"]
                )
                source_vector = np.array([[
                    rotate_point(anchors["right"], 120, 100, rotation)[0]
                    - rotate_point(anchors["left"], 120, 100, rotation)[0],
                    rotate_point(anchors["right"], 120, 100, rotation)[1]
                    - rotate_point(anchors["left"], 120, 100, rotation)[1],
                ]], dtype=np.float64)
                transformed_vector = source_vector @ transform[:2, :2].T
                self.assertGreater(transformed_vector[0, 0], 0)
                self.assertAlmostEqual(transformed_vector[0, 1], 0, delta=1e-4)
                self.assertGreaterEqual(rotated.shape[1], 100)
                self.assertGreaterEqual(rotated.shape[0], 100)

    def test_missing_anchors_do_not_claim_orientation(self):
        detections = [{
            "class": "ic",
            "cx": 30.0,
            "cy": 30.0,
            "confidence": 0.9,
        }]

        orientation, required, error = estimate_anchor_orientation(
            REFERENCE_COMPONENTS, detections, 0.25
        )

        self.assertTrue(required)
        self.assertIsNone(orientation)
        self.assertIsNotNone(error)

    def test_tight_crop_uses_reference_bounds_and_preserves_padding(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        orientation = {
            "left_cap": {"cx": 20.0, "cy": 20.0},
            "right_cap": {"cx": 60.0, "cy": 20.0},
            "ic": {"cx": 32.5, "cy": 14.0},
        }

        crop, transform, refined = crop_board_to_reference_bounds(
            image, REFERENCE_COMPONENTS, orientation, (80, 100, 3)
        )

        self.assertTrue(refined)
        self.assertLess(crop.shape[1], image.shape[1])
        self.assertLess(crop.shape[0], image.shape[0])
        reference_corners = np.float32([
            [0, 0], [99, 0], [99, 79], [0, 79],
        ]).reshape(-1, 1, 2)
        source_points = cv2.transform(
            reference_corners,
            cv2.invertAffineTransform(
                cv2.getAffineTransform(
                    np.float32([[20, 20], [60, 20], [32.5, 14]]),
                    np.float32([[10, 50], [90, 50], [35, 35]]),
                )
            ),
        )
        cropped_points = cv2.perspectiveTransform(source_points, transform)
        self.assertGreaterEqual(float(cropped_points[:, :, 0].min()), 0)
        self.assertGreaterEqual(float(cropped_points[:, :, 1].min()), 0)
        self.assertLess(float(cropped_points[:, :, 0].max()), crop.shape[1])
        self.assertLess(float(cropped_points[:, :, 1].max()), crop.shape[0])

    def test_tight_crop_is_skipped_when_reference_bounds_are_outside_frame(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        orientation = {
            "left_cap": {"cx": 20.0, "cy": 20.0},
            "right_cap": {"cx": 60.0, "cy": 20.0},
            "ic": {"cx": 32.5, "cy": 14.0},
        }

        crop, transform, refined = crop_board_to_reference_bounds(
            image, REFERENCE_COMPONENTS, orientation, (80, 100, 3)
        )

        self.assertFalse(refined)
        self.assertIs(crop, image)
        np.testing.assert_array_equal(transform, np.eye(3))


if __name__ == "__main__":
    unittest.main()
