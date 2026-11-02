import unittest

import cv2
import numpy as np

from pcb_crop_engine import PCBMonoCropper


class PCBComponentClusterCropTests(unittest.TestCase):
    def setUp(self):
        self.cropper = PCBMonoCropper(model=object())

    def test_crop_includes_all_component_edges_and_excludes_large_false_box(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        boxes = np.array([
            [12, 42, 34, 67],
            [38, 35, 63, 61],
            [67, 28, 94, 55],
            [99, 22, 126, 49],
            [131, 16, 158, 44],
            [0, 0, 200, 100],
        ], dtype=np.float32)

        cropped, info, rectangle = self.cropper.crop_component_cluster(
            image, boxes
        )

        self.assertIsNotNone(cropped)
        self.assertIsNotNone(info)
        self.assertIsNotNone(rectangle)
        self.assertGreaterEqual(cropped.shape[1], 145)
        self.assertLess(cropped.shape[1], image.shape[1])
        self.assertLess(cropped.shape[0], image.shape[0])

        component_corners = np.array([
            [12, 42], [34, 67], [38, 35], [63, 61],
            [67, 28], [94, 55], [99, 22], [126, 49],
            [131, 16], [158, 44],
        ], dtype=np.float32).reshape(-1, 1, 2)
        cropped_corners = cv2.perspectiveTransform(
            component_corners, info["transform"]
        )
        self.assertGreaterEqual(float(cropped_corners[:, :, 0].min()), -1)
        self.assertGreaterEqual(float(cropped_corners[:, :, 1].min()), -1)
        self.assertLess(float(cropped_corners[:, :, 0].max()), cropped.shape[1] + 1)
        self.assertLess(float(cropped_corners[:, :, 1].max()), cropped.shape[0] + 1)

    def test_tight_frame_edges_are_clamped_without_black_padding(self):
        image = np.full((165, 333, 3), 127, dtype=np.uint8)
        boxes = np.array([
            [4, 50, 70, 120],
            [78, 14, 154, 119],
            [163, 62, 259, 157],
            [267, 48, 332, 121],
            [75, 138, 124, 160],
        ], dtype=np.float32)

        cropped, info, _ = self.cropper.crop_component_cluster(image, boxes)

        self.assertIsNotNone(cropped)
        corners = info["corners"]
        self.assertGreaterEqual(float(corners[:, 0].min()), 0)
        self.assertGreaterEqual(float(corners[:, 1].min()), 0)
        self.assertLessEqual(float(corners[:, 0].max()), image.shape[1] - 1)
        self.assertLessEqual(float(corners[:, 1].max()), image.shape[0] - 1)
        self.assertEqual(int(cropped.min()), 127)
        self.assertEqual(int(cropped.max()), 127)

    def test_rotated_component_cluster_is_rectified_without_clipping(self):
        image = np.zeros((260, 320, 3), dtype=np.uint8)
        angle = np.deg2rad(32)
        along = np.array([np.cos(angle), np.sin(angle)])
        across = np.array([-np.sin(angle), np.cos(angle)])
        boxes = []
        component_centers = []
        for long_offset in (-65, 65):
            for short_offset in (-24, 24):
                center = np.array([160, 130]) + long_offset * along + short_offset * across
                component_centers.append(center)
                boxes.append([
                    center[0] - 6,
                    center[1] - 6,
                    center[0] + 6,
                    center[1] + 6,
                ])

        cropped, info, _ = self.cropper.crop_component_cluster(
            image, np.asarray(boxes, dtype=np.float32)
        )

        self.assertIsNotNone(cropped)
        self.assertGreater(cropped.shape[1], cropped.shape[0])
        transformed_centers = cv2.perspectiveTransform(
            np.asarray(component_centers, dtype=np.float32).reshape(-1, 1, 2),
            info["transform"],
        ).reshape(-1, 2)
        self.assertGreaterEqual(float(transformed_centers.min()), -1)
        self.assertLess(float(transformed_centers[:, 0].max()), cropped.shape[1] + 1)
        self.assertLess(float(transformed_centers[:, 1].max()), cropped.shape[0] + 1)

    def test_requires_multiple_component_detections(self):
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        cropped, info, rectangle = self.cropper.crop_component_cluster(
            image, np.array([[10, 10, 30, 30], [40, 40, 60, 60]])
        )

        self.assertIsNone(cropped)
        self.assertIsNone(info)
        self.assertIsNone(rectangle)
if __name__ == "__main__":
    unittest.main()
