import unittest
from unittest.mock import Mock

from fastapi import HTTPException

from backend.app.api.v1.common import run_inspection
from backend.app.services.pcb_engine import (
    STATUS_FAULTY,
    STATUS_OK,
    STATUS_REVIEW,
    evaluate_board_status,
    evaluate_component_inventory,
    match_components,
)


class ReferenceMatchingTests(unittest.TestCase):
    @staticmethod
    def detection(class_name, center, box, confidence=0.8):
        return {
            "class": class_name,
            "confidence": confidence,
            "ref_x": center[0],
            "ref_y": center[1],
            "ref_bbox": box,
        }

    def test_inventory_compares_total_count_even_if_detected_classes_differ(self):
        reference = [
            {"class": "resistor"},
            {"class": "resistor"},
            {"class": "capacitor"},
        ]
        detections = [
            {"class": "resistor"},
            {"class": "inductor"},
            {"class": "inductor"},
        ]

        status, stats, comparison = evaluate_component_inventory(reference, detections)

        self.assertEqual(status, "OK")
        self.assertEqual(stats["matched"], 3)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(stats["extra"], 0)
        self.assertEqual(
            comparison,
            [{
                "class": "Tổng số linh kiện",
                "reference_count": 3,
                "detected_count": 3,
                "matched": 3,
                "missing": 0,
                "extra": 0,
            }],
        )

    def test_one_missing_component_is_faulty_by_total_count(self):
        reference = [{"class": "resistor"} for _ in range(10)]
        reference.append({"class": "capacitor"})

        status, stats, comparison = evaluate_component_inventory(
            reference,
            [{"class": "resistor"} for _ in range(10)],
        )

        self.assertEqual(status, STATUS_FAULTY)
        self.assertEqual(stats["matched"], 10)
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(stats["extra"], 0)
        self.assertEqual(comparison[0]["reference_count"], 11)
        self.assertEqual(comparison[0]["detected_count"], 10)

    def test_more_than_reference_count_is_faulty(self):
        status, stats, _ = evaluate_component_inventory(
            [{"class": "resistor"}],
            [{"class": "resistor"}, {"class": "capacitor"}],
        )

        self.assertEqual(status, STATUS_FAULTY)
        self.assertEqual(stats["extra"], 1)

    def test_aligned_verdict_ignores_low_confidence_unmatched_detection(self):
        reference = [
            {
                "id": index + 1,
                "class": "resistor",
                "x": 100.0 + index * 100,
                "y": 100.0,
                "box_width": 20.0,
                "box_height": 20.0,
            }
            for index in range(11)
        ]
        detections = [
            self.detection(
                "resistor",
                (component["x"], component["y"]),
                (
                    component["x"] - 10,
                    component["y"] - 10,
                    component["x"] + 10,
                    component["y"] + 10,
                ),
            )
            for component in reference
        ]
        detections.append(
            self.detection("resistor", (178.0, 551.0), (168.0, 541.0, 188.0, 561.0), 0.37)
        )

        matches, _, used_references = match_components(
            reference, detections, tolerance=25.0
        )
        status, stats = evaluate_board_status(
            reference, detections, matches, used_references, aligned=True
        )
        (
            inventory_status,
            inventory_stats,
            inventory_comparison,
        ) = evaluate_component_inventory(reference, detections)

        self.assertEqual(status, STATUS_OK)
        self.assertEqual(stats["matched"], 11)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(stats["extra"], 0)
        self.assertEqual(inventory_status, STATUS_FAULTY)
        self.assertEqual(inventory_stats["extra"], 1)
        self.assertEqual(inventory_comparison[0]["detected_count"], 12)

        detections.extend(
            self.detection(
                "resistor",
                (x, 2500.0),
                (x - 10, 2490.0, x + 10, 2510.0),
                0.8,
            )
            for x in (2500.0, 3500.0, 4500.0)
        )
        matches, _, used_references = match_components(
            reference, detections, tolerance=25.0
        )
        status, stats = evaluate_board_status(
            reference, detections, matches, used_references, aligned=True
        )
        self.assertEqual(status, STATUS_REVIEW)
        self.assertEqual(stats["extra"], 3)

    def test_inventory_class_aliases_are_counted_together(self):
        status, stats, comparison = evaluate_component_inventory(
            [{"class": "IC"}],
            [{"class": "ic"}],
        )

        self.assertEqual(status, "OK")
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(stats["extra"], 0)
        self.assertEqual(comparison[0]["detected_count"], 1)

    def test_all_eleven_reference_components_are_standard(self):
        reference = [{"class": "resistor"} for _ in range(10)]
        reference.append({"class": "capacitor"})

        status, stats, _ = evaluate_component_inventory(
            reference,
            [{"class": "resistor"} for _ in range(10)]
            + [{"class": "capacitor"}],
        )

        self.assertEqual(status, "OK")
        self.assertEqual(stats["matched"], 11)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(stats["extra"], 0)

    def test_equal_total_is_standard_even_when_one_detected_class_differs(self):
        reference = [{"class": "resistor"} for _ in range(10)]
        reference.append({"class": "capacitor"})

        status, stats, _ = evaluate_component_inventory(
            reference,
            [{"class": "resistor"} for _ in range(10)]
            + [{"class": "inductor"}],
        )

        self.assertEqual(status, "OK")
        self.assertEqual(stats["matched"], 11)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(stats["extra"], 0)

    def test_wrong_class_at_reference_location_is_flagged_for_review(self):
        reference = [
            {
                "id": 1,
                "class": "inductor",
                "x": 100.0,
                "y": 100.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            self.detection("capacitor", (105.0, 105.0), (80.0, 80.0, 120.0, 120.0))
        ]

        matches, matched_detections, used_references = match_components(
            reference,
            detections,
            tolerance=100.0,
            allow_location_class_mismatch=True,
        )
        status, stats = evaluate_board_status(
            reference, detections, matches, used_references, aligned=True
        )

        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0]["class_match"])
        self.assertEqual(matched_detections, {0})
        self.assertEqual(used_references, {0})
        self.assertEqual(status, STATUS_REVIEW)
        self.assertEqual(stats["class_mismatch"], 1)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(stats["extra"], 0)

    def test_wrong_class_near_reference_matches_even_when_box_misses_reference_center(self):
        reference = [
            {
                "id": 10,
                "class": "inductor",
                "x": 100.0,
                "y": 100.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            self.detection(
                "capacitor",
                (112.0, 100.0),
                (108.0, 96.0, 116.0, 104.0),
            )
        ]

        matches, matched_detections, used_references = match_components(
            reference,
            detections,
            tolerance=100.0,
            allow_location_class_mismatch=True,
        )

        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0]["class_match"])
        self.assertEqual(matched_detections, {0})
        self.assertEqual(used_references, {0})

    def test_wrong_class_box_containing_reference_center_matches_for_review(self):
        reference = [
            {
                "id": 11,
                "class": "resistor",
                "x": 270.03,
                "y": 611.2,
                "box_width": 51.57,
                "box_height": 91.69,
            }
        ]
        detections = [
            self.detection(
                "capacitor",
                (274.04, 643.61),
                (251.29, 602.03, 296.79, 685.19),
            )
        ]

        matches, matched_detections, used_references = match_components(
            reference,
            detections,
            tolerance=73.26,
            allow_location_class_mismatch=True,
        )
        status, stats = evaluate_board_status(
            reference, detections, matches, used_references, aligned=True
        )

        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0]["class_match"])
        self.assertEqual(matched_detections, {0})
        self.assertEqual(used_references, {0})
        self.assertEqual(status, STATUS_REVIEW)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(stats["extra"], 0)

    def test_exact_class_match_is_preferred_over_location_fallback(self):
        reference = [
            {
                "class": "capacitor",
                "x": 100.0,
                "y": 100.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            self.detection("capacitor", (115.0, 105.0), (80.0, 80.0, 130.0, 130.0)),
            self.detection("resistor", (100.0, 100.0), (80.0, 80.0, 120.0, 120.0)),
        ]

        matches, _, _ = match_components(
            reference,
            detections,
            tolerance=100.0,
            allow_location_class_mismatch=True,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["det_index"], 0)
        self.assertTrue(matches[0]["class_match"])

    def test_distant_wrong_class_detection_does_not_hide_missing_component(self):
        reference = [
            {
                "class": "resistor",
                "x": 100.0,
                "y": 100.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            self.detection("capacitor", (140.0, 100.0), (130.0, 90.0, 150.0, 110.0))
        ]

        matches, _, used_references = match_components(
            reference,
            detections,
            tolerance=100.0,
            allow_location_class_mismatch=True,
        )
        status, stats = evaluate_board_status(
            reference, detections, matches, used_references, aligned=True
        )

        self.assertEqual(matches, [])
        self.assertEqual(status, STATUS_FAULTY)
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(stats["extra"], 1)

    def test_one_missing_component_is_faulty_even_when_allowance_is_nonzero(self):
        reference = [
            {"class": "resistor", "x": 100.0, "y": 100.0},
            {"class": "capacitor", "x": 200.0, "y": 100.0},
        ]
        detections = [
            self.detection("resistor", (100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        ]
        matches = [{"det_index": 0, "ref_index": 0, "class_match": True}]

        status, stats = evaluate_board_status(
            reference,
            detections,
            matches,
            used_references={0},
            aligned=True,
            missing_allowed=1,
        )

        self.assertEqual(status, STATUS_FAULTY)
        self.assertEqual(stats["missing"], 1)

    def test_complete_matching_board_remains_ok(self):
        reference = [{"class": "resistor", "x": 100.0, "y": 100.0}]
        detections = [
            self.detection("resistor", (100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        ]
        matches = [{"det_index": 0, "ref_index": 0, "class_match": True}]

        status, stats = evaluate_board_status(
            reference,
            detections,
            matches,
            used_references={0},
            aligned=True,
        )

        self.assertEqual(status, "OK")
        self.assertEqual(stats["missing"], 0)

    def test_large_box_does_not_match_when_detection_center_is_outside_tolerance(self):
        reference = [
            {
                "class": "capacitor",
                "x": 100.0,
                "y": 100.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            self.detection(
                "capacitor",
                (150.0, 100.0),
                (80.0, 80.0, 180.0, 120.0),
            )
        ]

        matches, _, used_references = match_components(
            reference, detections, tolerance=20.0
        )

        self.assertEqual(matches, [])
        self.assertEqual(used_references, set())

    def test_exact_class_match_obeys_strict_tolerance_even_when_box_contains_reference(self):
        reference = [
            {
                "class": "capacitor",
                "x": 100.0,
                "y": 100.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            self.detection(
                "capacitor",
                (125.0, 100.0),
                (80.0, 80.0, 140.0, 120.0),
            )
        ]

        matches, _, _ = match_components(reference, detections, tolerance=20.0)

        self.assertEqual(matches, [])

    def test_large_reference_component_gets_bounded_size_adaptive_tolerance(self):
        reference = [
            {
                "id": 10,
                "class": "inductor",
                "x": 100.0,
                "y": 100.0,
                "box_width": 396.0,
                "box_height": 219.0,
            }
        ]
        near_detection = [
            self.detection(
                "inductor",
                (190.0, 100.0),
                (0.0, 0.0, 396.0, 219.0),
            )
        ]
        far_detection = [
            self.detection(
                "inductor",
                (220.0, 100.0),
                (0.0, 0.0, 396.0, 219.0),
            )
        ]

        near_matches, _, _ = match_components(
            reference, near_detection, tolerance=73.0
        )
        far_matches, _, _ = match_components(
            reference, far_detection, tolerance=73.0
        )

        self.assertEqual(len(near_matches), 1)
        self.assertEqual(far_matches, [])

    def test_api_rejects_old_excessive_tolerance_value(self):
        pipeline = Mock(
            reference_image=object(),
            reference_components=[{"class": "capacitor"}],
        )

        with self.assertRaises(HTTPException) as raised:
            run_inspection(
                pipeline,
                frame=None,
                values={"tolerance_ratio": 0.55},
            )

        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
