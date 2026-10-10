import numpy as np

from backend.app.services.pcb_engine import (
    OrbAligner,
    estimate_component_alignment,
    find_board_corners,
    find_board_roi,
    rectify_board,
)


def align_board(reference_image: np.ndarray, scan_image: np.ndarray) -> np.ndarray | None:
    """Estimate a scan-to-reference homography from ORB features."""
    return OrbAligner(reference_image).compute_homography(scan_image)
