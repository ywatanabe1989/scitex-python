#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_utils/_auto_layout.py

"""Auto layout utilities for FTS elements."""

from typing import Any, Dict, List, Optional, Tuple

from ._calc_bounds import content_bounds
from ._normalize import Position, Size, normalize_position, normalize_size


def auto_layout_grid(
    n_elements: int,
    container_size: Size,
    margin_mm: float = 5.0,
    spacing_mm: float = 5.0,
    max_cols: Optional[int] = None,
) -> List[Tuple[Position, Size]]:
    """Generate grid layout positions and sizes for n elements.

    Args:
        n_elements: Number of elements to layout
        container_size: Container size
        margin_mm: Margin from container edges
        spacing_mm: Spacing between elements
        max_cols: Maximum columns (None = auto)

    Returns:
        List of (position, size) tuples for each element

    Example:
        >>> layouts = auto_layout_grid(4, {"width_mm": 170, "height_mm": 120})
        >>> len(layouts)
        4
    """
    if n_elements <= 0:
        return []

    container = normalize_size(container_size)

    # Determine grid dimensions
    if max_cols is None:
        cols = min(n_elements, 2)  # Default: max 2 columns
    else:
        cols = min(n_elements, max_cols)
    rows = (n_elements + cols - 1) // cols

    # Calculate element size
    available_width = container["width_mm"] - 2 * margin_mm - (cols - 1) * spacing_mm
    available_height = container["height_mm"] - 2 * margin_mm - (rows - 1) * spacing_mm
    elem_width = available_width / cols
    elem_height = available_height / rows

    results = []
    for i in range(n_elements):
        row = i // cols
        col = i % cols

        pos: Position = {
            "x_mm": margin_mm + col * (elem_width + spacing_mm),
            "y_mm": margin_mm + row * (elem_height + spacing_mm),
        }
        size: Size = {
            "width_mm": elem_width,
            "height_mm": elem_height,
        }
        results.append((pos, size))

    return results


def auto_crop_layout(
    elements: List[Dict[str, Any]],
    margin_mm: float = 5.0,
) -> Tuple[List[Dict[str, Any]], Size]:
    """Calculate auto-cropped layout by shifting elements and resizing canvas.

    This function:
    1. Finds the bounding box of all elements
    2. Shifts all positions so content starts at (margin, margin)
    3. Calculates new canvas size to fit content + margin

    Args:
        elements: List of element specifications (not modified in place)
        margin_mm: Margin to add around content (default: 5mm)

    Returns:
        Tuple of (shifted_elements, new_canvas_size)
        - shifted_elements: New list with adjusted positions
        - new_canvas_size: {"width_mm", "height_mm"} for the cropped canvas
    """
    if not elements:
        return [], {"width_mm": margin_mm * 2, "height_mm": margin_mm * 2}

    # Calculate content bounding box
    bounds = content_bounds(elements)
    if bounds is None:
        return [], {"width_mm": margin_mm * 2, "height_mm": margin_mm * 2}

    # Calculate offset to shift content to (margin, margin)
    offset_x = bounds["x_mm"] - margin_mm
    offset_y = bounds["y_mm"] - margin_mm

    # Shift all element positions
    shifted_elements = []
    for elem in elements:
        shifted = elem.copy()
        pos = normalize_position(elem.get("position"))
        shifted["position"] = {
            "x_mm": pos["x_mm"] - offset_x,
            "y_mm": pos["y_mm"] - offset_y,
        }
        shifted_elements.append(shifted)

    # Calculate new canvas size
    new_size: Size = {
        "width_mm": bounds["width_mm"] + margin_mm * 2,
        "height_mm": bounds["height_mm"] + margin_mm * 2,
    }

    return shifted_elements, new_size


__all__ = ["auto_layout_grid", "auto_crop_layout"]

# EOF
