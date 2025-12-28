#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_utils/_normalize.py

"""Normalization utilities for position and size values."""

from typing import Any, Dict, Optional

# Type aliases
Position = Dict[str, float]  # {"x_mm": float, "y_mm": float}
Size = Dict[str, float]  # {"width_mm": float, "height_mm": float}

# Default values
DEFAULT_POSITION: Position = {"x_mm": 0.0, "y_mm": 0.0}
DEFAULT_SIZE: Size = {"width_mm": 80.0, "height_mm": 60.0}


def normalize_position(pos: Optional[Dict[str, Any]]) -> Position:
    """Normalize position to standard format {"x_mm", "y_mm"}.

    Handles various input formats:
    - {"x_mm": 10, "y_mm": 20}  -> as-is
    - {"x": 10, "y": 20}        -> convert keys
    - None                      -> default (0, 0)

    Args:
        pos: Position dict or None

    Returns:
        Normalized position with x_mm and y_mm keys
    """
    if pos is None:
        return DEFAULT_POSITION.copy()

    return {
        "x_mm": float(pos.get("x_mm", pos.get("x", 0.0))),
        "y_mm": float(pos.get("y_mm", pos.get("y", 0.0))),
    }


def normalize_size(size: Optional[Dict[str, Any]]) -> Size:
    """Normalize size to standard format {"width_mm", "height_mm"}.

    Handles various input formats:
    - {"width_mm": 80, "height_mm": 60}  -> as-is
    - {"width": 80, "height": 60}        -> convert keys
    - None                               -> default size

    Args:
        size: Size dict or None

    Returns:
        Normalized size with width_mm and height_mm keys
    """
    if size is None:
        return DEFAULT_SIZE.copy()

    return {
        "width_mm": float(size.get("width_mm", size.get("width", 80.0))),
        "height_mm": float(size.get("height_mm", size.get("height", 60.0))),
    }


__all__ = [
    "normalize_position",
    "normalize_size",
    "Position",
    "Size",
    "DEFAULT_POSITION",
    "DEFAULT_SIZE",
]

# EOF
