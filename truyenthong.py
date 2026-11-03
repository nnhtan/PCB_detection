"""Standalone helper for sending an inspection verdict and resuming the conveyor."""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.app.services.conveyor import ConveyorClient


def send_inspection_result(
    is_ok: bool,
    port: str | None = None,
    baudrate: int | None = None,
    timeout: float = 5.0,
) -> str:
    """Send the verdict, resume the conveyor, and return the acknowledged verdict."""
    client = ConveyorClient(
        port=(
            port
            if port is not None
            else os.getenv("PCB_SERIAL_PORT", "COM5")
        ),
        baudrate=(
            baudrate
            if baudrate is not None
            else int(os.getenv("PCB_SERIAL_BAUDRATE", "115200"))
        ),
        timeout=timeout,
    )
    try:
        client.connect()
        verdict = client.send_result(is_ok)
        client.resume()
        return verdict
    finally:
        client.close()
