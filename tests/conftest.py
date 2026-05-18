"""Pytest path setup for article companion code."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_src = ROOT / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
