#!/usr/bin/env python3
"""CLI wrapper around the Shared Perception multi-camera loop."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    raise SystemExit(main())
