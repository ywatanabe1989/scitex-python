#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_utils/_calc_bounds.py

"""Bounding box calculation utilities for FTS elements."""

from typing import Any, Dict, List, Optional

from ._normalize import Size, normalize_position, normalize_size

# Type alias
Bounds = Dict[str, float]  # {"x_mm", "y_mm", "width_mm", "height_mm"}


def element_bounds(element: Dict[str, Any]) -> Bounds:
    """Get element bounding box.

    Args:
        element: Element specification with position and size

    Returns:
        Bounding box {x_mm, y_mm, width_mm, height_mm}
    """
    pos = normalize_position(element.get("position"))
    size = normalize_size(element.get("size"))
    return {
        "x_mm": pos["x_mm"],
        "y_mm": pos["y_mm"],
        "width_mm": size["width_mm"],
        "height_mm": size["height_mm"],
    }


def content_bounds(elements: List[Dict[str, Any]]) -> Optional[Bounds]:
    """Calculate the bounding box containing all elements.

    Args:
        elements: List of element specifications

    Returns:
        Bounding box {"x_mm", "y_mm", "width_mm", "height_mm"}
        containing all elements, or None if no elements.
    """
    if not elements:
        return None

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for elem in elements:
        bounds = element_bounds(elem)

        # Update min/max coordinates
        min_x = min(min_x, bounds["x_mm"])
        min_y = min(min_y, bounds["y_mm"])
        max_x = max(max_x, bounds["x_mm"] + bounds["width_mm"])
        max_y = max(max_y, bounds["y_mm"] + bounds["height_mm"])

    return {
        "x_mm": min_x,
        "y_mm": min_y,
        "width_mm": max_x - min_x,
        "height_mm": max_y - min_y,
    }


def validate_within_bounds(
    element: Dict[str, Any],
    container_size: Size,
    raise_error: bool = False,
) -> bool:
    """Check if element fits within container bounds.

    Args:
        element: Element with position and size
        container_size: Container size {"width_mm", "height_mm"}
        raise_error: If True, raise ValueError on violation

    Returns:
        True if element fits within container

    Raises:
        ValueError: If raise_error=True and element exceeds bounds
    """
    bounds = element_bounds(element)
    container = normalize_size(container_size)

    # Check bounds
    fits = (
        bounds["x_mm"] >= 0
        and bounds["y_mm"] >= 0
        and bounds["x_mm"] + bounds["width_mm"] <= container["width_mm"]
        and bounds["y_mm"] + bounds["height_mm"] <= container["height_mm"]
    )

    if not fits and raise_error:
        raise ValueError(
            f"Element exceeds container bounds: "
            f"element=({bounds['x_mm']}, {bounds['y_mm']}, "
            f"{bounds['width_mm']}x{bounds['height_mm']}), "
            f"container=({container['width_mm']}x{container['height_mm']})"
        )

    return fits


__all__ = ["element_bounds", "content_bounds", "validate_within_bounds", "Bounds"]

# EOF
