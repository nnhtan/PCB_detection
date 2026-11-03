from threading import RLock

import numpy as np
import pytest
from fastapi import HTTPException

from backend.app.api.v1 import camera
from backend.app.api.v1.common import run_tilted_inspection
from backend.app.services.camera_stream import CameraManager, FRAME_HISTORY_SIZE
from backend.app.services import pcb_engine


class FakePipeline:
    def __init__(self):
        self.reference_image = np.zeros((20, 30, 3), dtype=np.uint8)
        self.reference_components = [{"class": "ic"}]
        self.aligner = object()
        self.operation_lock = RLock()
        self.inspection_options = None

    def inspect_pcb(self, **options):
        self.inspection_options = options
        return {"crop_method": "full"}


def test_tilted_inspection_prefers_reference_alignment(monkeypatch):
    pipeline = FakePipeline()
    frame = np.ones((40, 50, 3), dtype=np.uint8)
    aligned_frame = np.full((20, 30, 3), 2, dtype=np.uint8)
    transform = np.eye(3, dtype=np.float64)
    monkeypatch.setattr(
        pcb_engine,
        "find_board_corners",
        lambda *args, **kwargs: pytest.fail("contour should only be a fallback"),
    )
    monkeypatch.setattr(
        pcb_engine,
        "find_reference_board_corners",
        lambda scan, aligner: np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    )
    monkeypatch.setattr(
        pcb_engine,
        "rectify_board_to_reference",
        lambda scan, corners, shape: (aligned_frame, transform),
    )

    result = run_tilted_inspection(pipeline, frame, {"conf": 0.4})

    assert pipeline.inspection_options["frame"] is aligned_frame
    assert pipeline.inspection_options["auto_crop"] is False
    assert pipeline.inspection_options["prealigned"] is True
    assert pipeline.inspection_options["conf"] == 0.4
    assert result["crop_method"] == "tilted_reference_orb"
    assert result["scan_mode"] == "tilted_reference"
    assert result["scan_alignment_transform"] == transform.tolist()


def test_tilted_inspection_falls_back_to_contour_and_orders_long_edge(monkeypatch):
    pipeline = FakePipeline()
    frame = np.ones((40, 50, 3), dtype=np.uint8)
    aligned_frame = np.full((20, 30, 3), 2, dtype=np.uint8)
    corners = np.float32([[0, 0], [0, 10], [30, 10], [30, 0]])
    transform = np.eye(3, dtype=np.float64)
    monkeypatch.setattr(
        pcb_engine, "find_reference_board_corners", lambda *args: None
    )
    monkeypatch.setattr(
        pcb_engine, "find_board_corners", lambda *args, **kwargs: corners
    )
    monkeypatch.setattr(
        pcb_engine,
        "rectify_board_to_reference",
        lambda scan, ordered, shape: (
            aligned_frame,
            transform,
        ),
    )

    result = run_tilted_inspection(pipeline, frame, {})

    assert pipeline.inspection_options["prealigned"] is False
    assert result["crop_method"] == "tilted_reference_contour"


def test_tilted_inspection_rejects_unmatched_scan_before_detection(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(pcb_engine, "find_board_corners", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pcb_engine, "find_reference_board_corners", lambda scan, aligner: None
    )

    with pytest.raises(HTTPException) as error:
        run_tilted_inspection(pipeline, np.zeros((40, 50, 3)), {})

    assert error.value.status_code == 422
    assert "Không tìm thấy PCB" in error.value.detail
    assert pipeline.inspection_options is None


def test_tilted_inspection_requires_reference():
    pipeline = FakePipeline()
    pipeline.reference_image = None
    pipeline.reference_components = []

    with pytest.raises(HTTPException) as error:
        run_tilted_inspection(pipeline, np.zeros((40, 50, 3)), {})

    assert error.value.status_code == 409


def test_camera_scan_modes_select_the_matching_inspection_flow():
    assert camera.CAMERA_FAST_IMGSZ == 1024
    assert camera.CAMERA_TILTED_IMGSZ == 1536
    assert camera._get_camera_imgsz("standard", None) == 1024
    assert camera._get_camera_imgsz("tilted_reference", None) == 1536
    assert camera._get_camera_imgsz("tilted_reference", 1024) == 1024
    assert camera._get_inspection_runner("standard") is camera.run_inspection
    assert (
        camera._get_inspection_runner("tilted_reference")
        is camera.run_tilted_inspection
    )
    with pytest.raises(HTTPException) as error:
        camera._get_inspection_runner("unsupported")
    assert error.value.status_code == 422


def test_tilted_camera_scan_selects_frame_with_most_board_clearance(monkeypatch):
    latest = np.full((40, 50, 3), 1, dtype=np.uint8)
    older = np.full((40, 50, 3), 2, dtype=np.uint8)
    monkeypatch.setattr(
        camera,
        "find_board_corners",
        lambda frame, **kwargs: (
            np.float32([[2, 2], [48, 2], [48, 38], [2, 38]])
            if frame[0, 0, 0] == 1
            else np.float32([[10, 10], [40, 10], [40, 30], [10, 30]])
        ),
    )

    selected_frame, selected_frame_id = camera._select_tilted_scan_frame(
        latest,
        20,
        [(latest, 20), (older, 19)],
        1.5,
    )

    assert selected_frame is older
    assert selected_frame_id == 19


def test_camera_manager_returns_recent_frames_newest_first_as_copies():
    manager = CameraManager()
    for frame_id in range(FRAME_HISTORY_SIZE + 2):
        manager.frame_history.append(
            (frame_id, np.full((2, 2, 3), frame_id, dtype=np.uint8))
        )

    frames = manager.get_recent_frames()

    assert [frame_id for _, frame_id in frames] == list(
        range(FRAME_HISTORY_SIZE + 1, 1, -1)
    )
    frames[0][0][0, 0] = 0
    assert manager.frame_history[-1][1][0, 0, 0] == FRAME_HISTORY_SIZE + 1
