from typing import Any

from backend.app.services.pcb_engine import (
    evaluate_board_status,
    match_components,
)


def compare_with_reference(
    detections: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    tolerance: float,
    missing_allowed: int = 0,
    extra_allowed: int = 2,
    conf_thresh: float = 0.25,
) -> dict[str, Any]:
    matches, _, used_references = match_components(
        reference, detections, tolerance, require_same_class=True
    )
    status, stats = evaluate_board_status(
        reference,
        detections,
        matches,
        used_references,
        aligned=True,
        missing_allowed=missing_allowed,
        extra_allowed=extra_allowed,
        conf_thresh=conf_thresh,
    )
    return {
        "status": status,
        "stats": stats,
        "matches": matches,
        "used_references": used_references,
    }
