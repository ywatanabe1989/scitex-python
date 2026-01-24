#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap/_constants.py

"""
Constants and type conversion utilities for hitmap generation.
"""

from typing import Any

import numpy as np

# Reserved colors for hitmap (human-readable)
HITMAP_BACKGROUND_COLOR = "#1a1a1a"  # Dark gray (not pure black, easier to see)
HITMAP_AXES_COLOR = "#404040"  # Medium gray (non-selectable axes elements)


def to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


__all__ = [
    "HITMAP_BACKGROUND_COLOR",
    "HITMAP_AXES_COLOR",
    "to_native",
]


# EOF
