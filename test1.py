"""Compatibility launcher for the unified PCB camera program in test.py."""

import importlib.util
import sys
from pathlib import Path


_MAIN_SCRIPT = Path(__file__).resolve().with_name("test.py")
_SPEC = importlib.util.spec_from_file_location("pcb_detection_test", _MAIN_SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Không thể nạp chương trình PCB: {_MAIN_SCRIPT}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


if __name__ == "__main__":
    _MODULE.main()
