#!/usr/bin/env python3
# Timestamp: "2025-12-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/layout.py

"""
Coordinate System and Layout Utilities for .stx Bundles.

Coordinate System:
- Origin (0,0) is at TOP-LEFT corner
- x_mm increases to the RIGHT
- y_mm increases DOWNWARD
- All positions are relative to parent's origin

    (0,0) ──────────────────► x_mm
      │
      │   ┌─────────────────────────┐
      │   │  Parent Bundle          │
      │   │   (0,0)                 │
      │   │    ├── Element A (10,10)│
      │   │    │    └── child (5,3) │  ← absolute: (15,13)
      │   │    └── Element B (90,10)│
      │   └─────────────────────────┘
      ▼
    y_mm

Usage:
    from scitex.fig.layout import to_absolute, normalize_position, element_bounds

    # Convert local to absolute coordinates
    abs_pos = to_absolute({"x_mm": 5, "y_mm": 3}, parent_pos={"x_mm": 10, "y_mm": 10})
    # Result: {"x_mm": 15, "y_mm": 13}

    # Get element bounding box
    bounds = element_bounds(element)
    # Result: {"x_mm": 10, "y_mm": 10, "width_mm": 70, "height_mm": 50}
"""

from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "to_absolute",
    "to_relative",
    "normalize_position",
    "normalize_size",
    "element_bounds",
    "content_bounds",
    "validate_within_bounds",
    "auto_layout_grid",
    "auto_crop_layout",
    "Position",
    "Size",
    "Bounds",
]

# Type aliases for clarity
Position = Dict[str, float]  # {"x_mm": float, "y_mm": float}
Size = Dict[str, float]  # {"width_mm": float, "height_mm": float}
Bounds = Dict[str, float]  # {"x_mm", "y_mm", "width_mm", "height_mm"}

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
        >>> layouts[0]
        ({"x_mm": 5.0, "y_mm": 5.0}, {"width_mm": 77.5, "height_mm": 52.5})
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


def content_bounds(elements: List[Dict[str, Any]]) -> Optional[Bounds]:
    """Calculate the bounding box containing all elements.

    Args:
        elements: List of element specifications

    Returns:
        Bounding box {"x_mm", "y_mm", "width_mm", "height_mm"}
        containing all elements, or None if no elements.

    Example:
        >>> elements = [
        ...     {"position": {"x_mm": 10, "y_mm": 20}, "size": {"width_mm": 50, "height_mm": 30}},
        ...     {"position": {"x_mm": 80, "y_mm": 10}, "size": {"width_mm": 40, "height_mm": 60}},
        ... ]
        >>> content_bounds(elements)
        {"x_mm": 10.0, "y_mm": 10.0, "width_mm": 110.0, "height_mm": 60.0}
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

    Example:
        >>> elements = [
        ...     {"id": "A", "position": {"x_mm": 50, "y_mm": 40}, "size": {"width_mm": 30, "height_mm": 20}},
        ...     {"id": "B", "position": {"x_mm": 100, "y_mm": 30}, "size": {"width_mm": 40, "height_mm": 50}},
        ... ]
        >>> shifted, new_size = auto_crop_layout(elements, margin_mm=5)
        >>> # Element A: was at (50, 40), content starts at (50, 30)
        >>> # Shift: (50-50+5, 30-30+5) = (5, 5)
        >>> shifted[0]["position"]
        {"x_mm": 5.0, "y_mm": 15.0}  # (50 - 50 + 5, 40 - 30 + 5)
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


# EOF
