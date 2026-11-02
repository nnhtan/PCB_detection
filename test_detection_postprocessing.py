import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from backend.app.services.pcb_engine import (
    PCBPipeline,
    normalize_component_class,
)


class DetectionPostprocessingTests(unittest.TestCase):
    @staticmethod
    def detection(class_name, confidence, box):
        return {"class_name": class_name, "confidence": confidence, "box": box}

    def test_mismatch_annotation_shows_reference_and_preserves_yolo_prediction(self):
        detection = {
            "class": "diode",
            "confidence": 0.73,
            "class_match": False,
            "matched_ref_class": "resistor",
            "status": "REVIEW",
        }

        label = PCBPipeline._format_detection_annotation_label(detection)

        self.assertEqual(
            label, "resistor (YOLO: diode 0.73) [REVIEW]"
        )
        self.assertEqual(detection["class"], "diode")
        self.assertEqual(detection["status"], "REVIEW")

    def test_reference_class_is_authoritative_but_yolo_prediction_is_preserved(self):
        reference = [{"id": 5, "class": "resistor"}]
        detections = [{"class": "LED", "confidence": 0.50}]
        matches = [
            {
                "ref_index": 0,
                "det_index": 0,
                "class_match": False,
            }
        ]

        PCBPipeline._apply_reference_class_authority(
            reference, detections, matches
        )

        self.assertEqual(detections[0]["class"], "resistor")
        self.assertEqual(detections[0]["yolo_class"], "LED")
        self.assertFalse(detections[0]["yolo_class_match"])
        self.assertTrue(detections[0]["class_match"])
        self.assertTrue(matches[0]["class_match"])
        self.assertFalse(matches[0]["yolo_class_match"])
        self.assertEqual(
            PCBPipeline._format_detection_annotation_label(
                {**detections[0], "status": "OK"}
            ),
            "resistor (YOLO: LED 0.50) [REF]",
        )

    def test_removes_unsupported_detection_classes(self):
        detections = [
            self.detection("buzzer", 0.404, (0, 0, 100, 100)),
            self.detection("capacitor", 0.90, (200, 0, 300, 100)),
        ]

        result = PCBPipeline._filter_unsupported_classes(detections)

        self.assertEqual([item["class_name"] for item in result], ["capacitor"])

    def test_normalizes_duplicate_model_labels_without_merging_component_types(self):
        detections = [
            self.detection("Capacitor", 0.90, (0, 0, 100, 100)),
            self.detection("Resistor", 0.80, (120, 0, 160, 40)),
            self.detection("inductor", 0.85, (200, 0, 300, 100)),
            self.detection("potentiometer", 0.82, (320, 0, 420, 100)),
        ]

        result = PCBPipeline._filter_unsupported_classes(detections)

        self.assertEqual(
            [item["class_name"] for item in result],
            ["capacitor", "resistor", "inductor", "potentiometer"],
        )
        self.assertNotEqual(
            normalize_component_class("inductor"),
            normalize_component_class("potentiometer"),
        )

    def test_merge_keeps_overlapping_different_class_hypotheses(self):
        full_frame = [
            self.detection("inductor", 0.589, (907, 12, 1284, 222)),
        ]
        sliced = [
            self.detection("switch", 0.674, (894, 7, 1082, 203)),
        ]

        result = PCBPipeline._merge_full_frame_and_sliced_detections(
            full_frame, sliced
        )

        self.assertEqual(
            [item["class_name"] for item in result],
            ["inductor", "switch"],
        )

    def test_keeps_slice_detection_without_full_frame_overlap(self):
        full_frame = [
            self.detection("capacitor", 0.90, (0, 0, 100, 100)),
        ]
        sliced = [
            self.detection("resistor", 0.80, (120, 0, 160, 40)),
        ]

        result = PCBPipeline._merge_full_frame_and_sliced_detections(
            full_frame, sliced
        )

        self.assertEqual(
            [item["class_name"] for item in result],
            ["capacitor", "resistor"],
        )

    def test_keeps_partially_overlapping_neighboring_components(self):
        detections = [
            self.detection("capacitor", 0.90, (0, 0, 100, 100)),
            self.detection("resistor", 0.80, (90, 0, 190, 100)),
        ]

        result = PCBPipeline._suppress_overlapping_detections(detections)

        self.assertEqual(len(result), 2)

    def test_keeps_overlapping_hypotheses_with_different_classes(self):
        detections = [
            self.detection("inductor", 0.90, (0, 0, 100, 100)),
            self.detection("potentiometer", 0.80, (20, 20, 80, 80)),
        ]

        result = PCBPipeline._suppress_overlapping_detections(detections)

        self.assertEqual(
            [item["class_name"] for item in result],
            ["inductor", "potentiometer"],
        )

    def test_keeps_only_highest_confidence_box_for_same_class(self):
        detections = [
            self.detection("inductor", 0.90, (0, 0, 100, 100)),
            self.detection("inductor", 0.80, (20, 20, 80, 80)),
        ]

        result = PCBPipeline._suppress_overlapping_detections(detections)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["confidence"], 0.90)

    def test_reference_roi_uses_low_confidence_scaled_prediction_and_maps_box(self):
        pipeline = PCBPipeline.__new__(PCBPipeline)
        pipeline.reference_components = [
            {
                "class": "resistor",
                "x": 200.0,
                "y": 200.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        pipeline.device = "cpu"
        pipeline.model = Mock()
        pipeline.model.names = {15: "resistor"}
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy = torch.tensor([[100.0, 100.0, 180.0, 180.0]])
        boxes.cls = torch.tensor([15.0])
        boxes.conf = torch.tensor([0.02])
        pipeline.model.predict.return_value = [
            SimpleNamespace(boxes=boxes)
        ]

        result = pipeline._detect_missing_reference_rois(
            np.zeros((400, 400, 3), dtype=np.uint8),
            [0],
            np.eye(3),
            imgsz=1536,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "resistor")
        self.assertAlmostEqual(result[0]["confidence"], 0.02)
        self.assertEqual(
            (result[0]["x1"], result[0]["y1"], result[0]["x2"], result[0]["y2"]),
            (190.0, 190.0, 230.0, 230.0),
        )
        prediction_kwargs = pipeline.model.predict.call_args.kwargs
        self.assertEqual(prediction_kwargs["conf"], 0.01)
        self.assertTrue(prediction_kwargs["augment"])
        self.assertFalse(prediction_kwargs["agnostic_nms"])

    def test_reference_roi_recovery_requires_expected_class_and_location(self):
        pipeline = PCBPipeline.__new__(PCBPipeline)
        pipeline.reference_components = [
            {
                "class": "resistor",
                "x": 200.0,
                "y": 200.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        roi_detections = [
            {
                "class": "resistor",
                "_roi_ref_index": 0,
                "cx": 202.0,
                "cy": 203.0,
                "x1": 180.0,
                "y1": 180.0,
                "x2": 224.0,
                "y2": 226.0,
            },
            {
                "class": "capacitor",
                "_roi_ref_index": 0,
                "cx": 200.0,
                "cy": 200.0,
                "x1": 180.0,
                "y1": 180.0,
                "x2": 220.0,
                "y2": 220.0,
            },
            {
                "class": "resistor",
                "_roi_ref_index": 0,
                "cx": 260.0,
                "cy": 200.0,
                "x1": 240.0,
                "y1": 180.0,
                "x2": 280.0,
                "y2": 220.0,
            },
        ]

        result = pipeline._filter_reference_roi_detections(
            roi_detections, [0], np.eye(3), tolerance=20.0
        )

        self.assertEqual(result, [roi_detections[0]])

    def test_reference_matching_class_suppresses_overlapping_alternative(self):
        reference = [
            {
                "class": "potentiometer",
                "x": 50.0,
                "y": 50.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            {
                "class": "potentiometer",
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (30.0, 30.0, 70.0, 70.0),
            },
            {
                "class": "inductor",
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (31.0, 31.0, 69.0, 69.0),
            },
        ]

        result = PCBPipeline._suppress_class_conflicts_by_reference(
            reference, detections, tolerance=100.0
        )

        self.assertEqual([item["class"] for item in result], ["potentiometer"])

    def test_reference_match_suppresses_larger_overlapping_class_hypothesis(self):
        reference = [
            {
                "class": "resistor",
                "x": 50.0,
                "y": 50.0,
                "box_width": 12.0,
                "box_height": 20.0,
            }
        ]
        detections = [
            {
                "class": "resistor",
                "confidence": 0.04,
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (45.0, 45.0, 55.0, 55.0),
            },
            {
                "class": "capacitor",
                "confidence": 0.45,
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (30.0, 30.0, 70.0, 70.0),
            },
        ]

        result = PCBPipeline._suppress_class_conflicts_by_reference(
            reference, detections, tolerance=100.0
        )

        self.assertEqual([item["class"] for item in result], ["resistor"])

    def test_reference_matching_does_not_suppress_another_exact_component(self):
        reference = [
            {
                "class": "resistor",
                "x": 50.0,
                "y": 50.0,
                "box_width": 30.0,
                "box_height": 30.0,
            },
            {
                "class": "capacitor",
                "x": 65.0,
                "y": 50.0,
                "box_width": 30.0,
                "box_height": 30.0,
            },
        ]
        detections = [
            {
                "class": "resistor",
                "confidence": 0.8,
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (35.0, 35.0, 65.0, 65.0),
            },
            {
                "class": "capacitor",
                "confidence": 0.8,
                "ref_x": 65.0,
                "ref_y": 50.0,
                "ref_bbox": (50.0, 35.0, 80.0, 65.0),
            },
        ]

        result = PCBPipeline._suppress_class_conflicts_by_reference(
            reference, detections, tolerance=100.0
        )

        self.assertEqual(
            [item["class"] for item in result],
            ["resistor", "capacitor"],
        )

    def test_reference_does_not_hide_wrong_class_when_no_matching_hypothesis(self):
        reference = [
            {
                "class": "potentiometer",
                "x": 50.0,
                "y": 50.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            {
                "class": "inductor",
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (30.0, 30.0, 70.0, 70.0),
            }
        ]

        result = PCBPipeline._suppress_class_conflicts_by_reference(
            reference, detections, tolerance=100.0
        )

        self.assertEqual(result, detections)

    def test_reference_keeps_best_wrong_class_hypothesis_when_no_exact_class_exists(self):
        reference = [
            {
                "class": "potentiometer",
                "x": 50.0,
                "y": 50.0,
                "box_width": 40.0,
                "box_height": 40.0,
            }
        ]
        detections = [
            {
                "class": "inductor",
                "confidence": 0.90,
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (30.0, 30.0, 70.0, 70.0),
            },
            {
                "class": "switch",
                "confidence": 0.80,
                "ref_x": 50.0,
                "ref_y": 50.0,
                "ref_bbox": (31.0, 31.0, 69.0, 69.0),
            },
        ]

        result = PCBPipeline._suppress_class_conflicts_by_reference(
            reference, detections, tolerance=100.0
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "inductor")

    def test_suppresses_detection_over_reference_mounting_hole(self):
        detection = {
            "class": "LED",
            "cx": 280.0,
            "cy": 148.0,
            "x1": 203.0,
            "y1": 67.0,
            "x2": 357.0,
            "y2": 229.0,
            "ref_x": 280.0,
            "ref_y": 148.0,
            "ref_bbox": (203.0, 67.0, 357.0, 229.0),
        }

        result = PCBPipeline._suppress_reference_hole_detections(
            [detection], [(269.0, 79.0, 61.0)]
        )

        self.assertEqual(result, [])

    def test_keeps_detection_away_from_reference_mounting_holes(self):
        detection = {
            "class": "resistor",
            "cx": 500.0,
            "cy": 400.0,
            "x1": 480.0,
            "y1": 380.0,
            "x2": 520.0,
            "y2": 420.0,
            "ref_x": 500.0,
            "ref_y": 400.0,
            "ref_bbox": (480.0, 380.0, 520.0, 420.0),
        }

        result = PCBPipeline._suppress_reference_hole_detections(
            [detection], [(269.0, 79.0, 61.0)]
        )

        self.assertEqual(result, [detection])

    def test_preserves_small_component_inside_large_detection_box(self):
        detections = [
            self.detection("ic", 0.95, (0, 0, 300, 300)),
            self.detection("resistor", 0.80, (10, 10, 30, 30)),
        ]

        result = PCBPipeline._suppress_overlapping_detections(detections)

        self.assertEqual(
            [item["class_name"] for item in result],
            ["ic", "resistor"],
        )

    def test_ic_filter_suppresses_component_detections_on_ic_pins(self):
        detections = [
            self.detection("ic", 0.95, (0, 0, 300, 300)),
            self.detection("transistor", 0.47, (30, 180, 90, 280)),
            self.detection("resistor", 0.40, (120, 200, 150, 260)),
        ]

        result = PCBPipeline._suppress_ic_artifacts(detections)

        self.assertEqual([item["class_name"] for item in result], ["ic"])

    def test_ic_filter_keeps_components_outside_ic_box(self):
        detections = [
            self.detection("ic", 0.95, (0, 0, 300, 300)),
            self.detection("capacitor", 0.80, (310, 100, 370, 180)),
        ]

        result = PCBPipeline._suppress_ic_artifacts(detections)

        self.assertEqual(
            [item["class_name"] for item in result],
            ["ic", "capacitor"],
        )

    def test_reference_creation_uses_postprocessed_detector_results(self):
        detection = {
            "id": 1,
            "class_id": 6,
            "class": "capacitor",
            "confidence": 0.91,
            "cx": 20.0,
            "cy": 30.0,
            "x1": 10.0,
            "y1": 15.0,
            "x2": 30.0,
            "y2": 45.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pipeline = PCBPipeline.__new__(PCBPipeline)
            pipeline.ref_image_path = str(root / "reference.jpg")
            pipeline.ref_json_path = str(root / "reference_components.json")
            pipeline.detect_components = Mock(return_value=[detection])
            pipeline.load_reference = Mock()

            result = pipeline.create_reference_from_image(
                np.zeros((60, 40, 3), dtype=np.uint8)
            )

            self.assertEqual(result["components_count"], 1)
            self.assertEqual(result["components"][0]["class"], "capacitor")
            saved_data = json.loads(
                Path(pipeline.ref_json_path).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_data["components"], result["components"])

    def test_keeps_slice_detection_inside_large_full_frame_box(self):
        full_frame = [
            self.detection("ic", 0.95, (0, 0, 300, 300)),
        ]
        sliced = [
            self.detection("resistor", 0.80, (10, 10, 30, 30)),
        ]

        result = PCBPipeline._merge_full_frame_and_sliced_detections(
            full_frame, sliced
        )

        self.assertEqual(
            [item["class_name"] for item in result],
            ["ic", "resistor"],
        )

    def test_accuracy_mode_uses_gpu_batching_and_test_time_augmentation(self):
        pipeline = PCBPipeline.__new__(PCBPipeline)
        pipeline.sahi_model = SimpleNamespace(image_size=0)
        pipeline.device = 0
        pipeline.use_cuda = True
        pipeline.model = Mock()
        pipeline.model.predict.return_value = [
            SimpleNamespace(boxes=None)
        ]
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        with patch(
            "backend.app.services.pcb_engine.get_sliced_prediction",
            return_value=SimpleNamespace(object_prediction_list=[]),
        ) as sliced_prediction:
            detections = pipeline.detect_components(frame)

        self.assertEqual(detections, [])
        self.assertTrue(pipeline.model.predict.call_args.kwargs["augment"])
        self.assertEqual(sliced_prediction.call_args.kwargs["batch_size"], 2)

    def test_fast_cpu_mode_keeps_single_batch_without_augmentation(self):
        pipeline = PCBPipeline.__new__(PCBPipeline)
        pipeline.sahi_model = SimpleNamespace(image_size=0)
        pipeline.device = "cpu"
        pipeline.use_cuda = False
        pipeline.model = Mock()
        pipeline.model.predict.return_value = [
            SimpleNamespace(boxes=None)
        ]
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        with patch(
            "backend.app.services.pcb_engine.get_sliced_prediction",
            return_value=SimpleNamespace(object_prediction_list=[]),
        ) as sliced_prediction:
            detections = pipeline.detect_components(frame, fast_mode=True)

        self.assertEqual(detections, [])
        self.assertFalse(pipeline.model.predict.call_args.kwargs["augment"])
        self.assertEqual(sliced_prediction.call_args.kwargs["batch_size"], 1)

if __name__ == "__main__":
    unittest.main()
