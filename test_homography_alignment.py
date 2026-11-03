import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.app.services.pcb_engine import (
    estimate_component_alignment,
    find_board_corners,
    find_reference_board_corners,
    order_board_corners_for_reference,
    rectify_board,
    rectify_board_to_reference,
    warp_board_to_reference,
)


class ComponentHomographyTests(unittest.TestCase):
    class FixedHomographyAligner:
        def __init__(self, transform, reference_shape):
            self.transform = transform
            self.ref_shape = reference_shape
            self.valid = True

        def compute_homography(self, frame):
            return self.transform

    def test_rectify_board_preserves_full_frame_edge_pixels(self):
        frame = np.arange(60 * 80 * 3, dtype=np.uint8).reshape((60, 80, 3))
        corners = np.float32([
            [0, 0],
            [79, 0],
            [79, 59],
            [0, 59],
        ])

        rectified, transform = rectify_board(frame, corners)

        self.assertEqual(rectified.shape, frame.shape)
        np.testing.assert_array_equal(rectified, frame)
        np.testing.assert_allclose(transform, np.eye(3), atol=1e-6)

    def test_rectify_board_uses_reference_resolution_and_orientation(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        corners = np.float32([
            [10, 10],
            [89, 5],
            [95, 70],
            [5, 75],
        ])

        rectified, transform = rectify_board_to_reference(
            frame, corners, (60, 120, 3)
        )

        self.assertEqual(rectified.shape, (60, 120, 3))
        expected_corners = np.float32([
            [0, 0],
            [119, 0],
            [119, 59],
            [0, 59],
        ]).reshape(-1, 1, 2)
        actual_corners = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), transform
        )
        np.testing.assert_allclose(actual_corners, expected_corners, atol=1e-3)

    def test_board_contour_keeps_long_edge_horizontal_across_tilt_angles(self):
        reference_aspect = 2.25
        background = np.full((600, 800, 3), 220, dtype=np.uint8)

        for angle in range(-60, 61, 5):
            frame = background.copy()
            rectangle = ((400, 300), (450, 200), angle)
            cv2.fillConvexPoly(
                frame,
                cv2.boxPoints(rectangle).astype(np.int32),
                (40, 80, 120),
            )
            corners = find_board_corners(
                frame,
                reference_aspect=reference_aspect,
                padding_ratio=0.0,
                include_dark_regions=True,
            )

            self.assertIsNotNone(corners, f"contour not found at {angle} degrees")
            ordered = order_board_corners_for_reference(
                corners, reference_aspect
            )
            first_edge = np.linalg.norm(ordered[1] - ordered[0])
            second_edge = np.linalg.norm(ordered[2] - ordered[1])
            self.assertGreater(first_edge, second_edge)

    def test_dark_board_is_found_when_background_edges_join_frame_border(self):
        frame = np.full((600, 800, 3), 235, dtype=np.uint8)
        board = cv2.boxPoints(((400, 300), (450, 200), 18)).astype(np.int32)
        cv2.fillConvexPoly(frame, board, (25, 25, 25))
        frame_edges = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(frame_edges, (0, 0), (639, 479), 255, 2)
        with patch("backend.app.services.pcb_engine.cv2.Canny") as canny:
            canny.return_value = frame_edges.copy()
            original_corners = find_board_corners(
                frame,
                reference_aspect=2.25,
                padding_ratio=0.0,
            )
            tilted_corners = find_board_corners(
                frame,
                reference_aspect=2.25,
                padding_ratio=0.0,
                include_dark_regions=True,
            )

        self.assertIsNone(original_corners)
        self.assertIsNotNone(tilted_corners)
        ordered = order_board_corners_for_reference(tilted_corners, 2.25)
        first_edge = np.linalg.norm(ordered[1] - ordered[0])
        second_edge = np.linalg.norm(ordered[2] - ordered[1])
        self.assertGreater(first_edge, second_edge)

    def test_alignment_warp_outputs_reference_sized_rectified_crop(self):
        frame = np.zeros((100, 140, 3), dtype=np.uint8)
        frame[40:60, 60:80] = (255, 255, 255)
        transform = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ])

        rectified = warp_board_to_reference(
            frame, transform, (50, 70, 3)
        )

        self.assertEqual(rectified.shape, (50, 70, 3))
        self.assertGreater(int(rectified[24:31, 29:41].max()), 0)

    def test_reference_rectification_preserves_180_degree_orientation(self):
        reference = np.arange(60 * 80 * 3, dtype=np.uint8).reshape((60, 80, 3))
        scan = cv2.rotate(reference, cv2.ROTATE_180)
        transform = np.array([
            [-1, 0, 79],
            [0, -1, 59],
            [0, 0, 1],
        ], dtype=np.float64)
        aligner = self.FixedHomographyAligner(transform, reference.shape)

        corners = find_reference_board_corners(scan, aligner)
        self.assertIsNotNone(corners)
        rectified, _ = rectify_board_to_reference(scan, corners, reference.shape)

        np.testing.assert_array_equal(rectified, reference)

    def test_component_alignment_recovers_perspective_with_an_outlier(self):
        reference_points = np.float32([
            [25, 30],
            [205, 25],
            [210, 205],
            [30, 210],
            [115, 115],
            [75, 135],
        ])
        expected_homography = np.array([
            [1.05, 0.08, -8.0],
            [-0.04, 0.98, 7.0],
            [0.0007, -0.0004, 1.0],
        ], dtype=np.float64)
        scan_points = cv2.perspectiveTransform(
            reference_points.reshape(-1, 1, 2),
            np.linalg.inv(expected_homography),
        ).reshape(-1, 2)
        scan_points[-1] += [35, -25]

        classes = ["ic", "capacitor", "diode", "resistor", "led", "transistor"]
        reference_components = [
            {"class": class_name, "x": float(point[0]), "y": float(point[1])}
            for class_name, point in zip(classes, reference_points)
        ]
        detections = [
            {
                "class": class_name,
                "cx": float(point[0]),
                "cy": float(point[1]),
                "confidence": 0.95,
            }
            for class_name, point in zip(classes, scan_points)
        ]

        transform = estimate_component_alignment(
            reference_components,
            detections,
            tolerance=10.0,
            conf_thresh=0.35,
            crop_shape=(240, 240, 3),
            ref_shape=(240, 240, 3),
        )

        self.assertIsNotNone(transform)
        self.assertFalse(np.allclose(transform[2], [0.0, 0.0, 1.0], atol=1e-6))
        aligned_points = cv2.perspectiveTransform(
            scan_points[:5].reshape(-1, 1, 2), transform
        ).reshape(-1, 2)
        np.testing.assert_allclose(aligned_points, reference_points[:5], atol=1.0)

    def test_component_alignment_keeps_affine_fallback_for_three_anchors(self):
        reference_points = np.float32([[25, 30], [205, 25], [115, 190]])
        affine = np.array([
            [1.1, 0.05, 8.0],
            [-0.03, 0.95, 12.0],
        ], dtype=np.float32)
        scan_points = cv2.transform(
            reference_points.reshape(-1, 1, 2), cv2.invertAffineTransform(affine)
        ).reshape(-1, 2)
        classes = ["ic", "capacitor", "diode"]
        reference_components = [
            {"class": class_name, "x": float(point[0]), "y": float(point[1])}
            for class_name, point in zip(classes, reference_points)
        ]
        detections = [
            {
                "class": class_name,
                "cx": float(point[0]),
                "cy": float(point[1]),
                "confidence": 0.95,
            }
            for class_name, point in zip(classes, scan_points)
        ]

        transform = estimate_component_alignment(
            reference_components,
            detections,
            tolerance=10.0,
            conf_thresh=0.35,
            crop_shape=(240, 240, 3),
            ref_shape=(240, 240, 3),
        )

        self.assertIsNotNone(transform)
        np.testing.assert_allclose(transform[2], [0.0, 0.0, 1.0], atol=1e-6)
        aligned_points = cv2.perspectiveTransform(
            scan_points.reshape(-1, 1, 2), transform
        ).reshape(-1, 2)
        np.testing.assert_allclose(aligned_points, reference_points, atol=1.0)


if __name__ == "__main__":
    unittest.main()
