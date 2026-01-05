#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_utils/_convert_coords.py

"""Coordinate conversion utilities for FTS layout.

Coordinate System:
- Origin (0,0) is at TOP-LEFT corner
- x_mm increases to the RIGHT
- y_mm increases DOWNWARD
"""

from typing import Dict, Optional

from ._normalize import normalize_position

# Type alias
Position = Dict[str, float]  # {"x_mm": float, "y_mm": float}


def to_absolute(
    local_pos: Position,
    parent_pos: Optional[Position] = None,
) -> Position:
    """Convert local position to absolute (figure-level) coordinates.

    Args:
        local_pos: Position relative to parent's origin
        parent_pos: Parent's absolute position (None = root level)

    Returns:
        Absolute position in figure coordinates

    Example:
        >>> to_absolute({"x_mm": 5, "y_mm": 3}, {"x_mm": 10, "y_mm": 10})
        {"x_mm": 15.0, "y_mm": 13.0}
    """
    local = normalize_position(local_pos)

    if parent_pos is None:
        return local

    parent = normalize_position(parent_pos)
    return {
        "x_mm": parent["x_mm"] + local["x_mm"],
        "y_mm": parent["y_mm"] + local["y_mm"],
    }


def to_relative(
    absolute_pos: Position,
    parent_pos: Position,
) -> Position:
    """Convert absolute position to local (parent-relative) coordinates.

    Args:
        absolute_pos: Absolute position in figure coordinates
        parent_pos: Parent's absolute position

    Returns:
        Position relative to parent's origin

    Example:
        >>> to_relative({"x_mm": 15, "y_mm": 13}, {"x_mm": 10, "y_mm": 10})
        {"x_mm": 5.0, "y_mm": 3.0}
    """
    absolute = normalize_position(absolute_pos)
    parent = normalize_position(parent_pos)
    return {
        "x_mm": absolute["x_mm"] - parent["x_mm"],
        "y_mm": absolute["y_mm"] - parent["y_mm"],
    }


__all__ = ["to_absolute", "to_relative", "Position"]

# EOF
