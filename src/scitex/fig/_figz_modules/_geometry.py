#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_geometry.py

"""Geometry extraction for Figz bundles (GUI editing support)."""

from typing import Any, Dict, List, Optional, Tuple

# Default DPI for rendering
DEFAULT_DPI = 150


def mm_to_px(mm: float, dpi: int = DEFAULT_DPI) -> float:
    """Convert millimeters to pixels at given DPI."""
    return mm * dpi / 25.4


def extract_geometry(
    elements: List[Dict[str, Any]],
    size_mm: Dict[str, float],
    dpi: int = DEFAULT_DPI,
    actual_size_px: Optional[Tuple[int, int]] = None,
) -> dict:
    """Extract geometry data for all elements (hit areas for GUI editing).

    Args:
        elements: List of element specifications
        size_mm: Canvas size {"width": mm, "height": mm}
        dpi: Resolution for pixel conversion
        actual_size_px: Actual rendered image size (width, height) in pixels.
                        If provided, uses this for accurate px conversion.

    Returns:
        Geometry data with element positions, sizes, and hit areas in both mm and px
    """
    canvas_w_mm = size_mm.get("width", 170)
    canvas_h_mm = size_mm.get("height", 120)

    # Calculate pixel dimensions
    if actual_size_px:
        canvas_w_px, canvas_h_px = actual_size_px
        # Calculate scale factors for accurate conversion
        scale_x = canvas_w_px / canvas_w_mm
        scale_y = canvas_h_px / canvas_h_mm
    else:
        # Use standard mm to px conversion
        canvas_w_px = int(mm_to_px(canvas_w_mm, dpi))
        canvas_h_px = int(mm_to_px(canvas_h_mm, dpi))
        scale_x = dpi / 25.4
        scale_y = dpi / 25.4

    geometry = {
        "canvas": {
            "width_mm": canvas_w_mm,
            "height_mm": canvas_h_mm,
            "width_px": canvas_w_px,
            "height_px": canvas_h_px,
            "dpi": dpi,
        },
        "elements": [],
    }

    for elem in elements:
        elem_geom = {
            "id": elem.get("id"),
            "type": elem.get("type"),
        }

        # Position
        pos = elem.get("position", {})
        x_mm = pos.get("x_mm", 0)
        y_mm = pos.get("y_mm", 0)
        elem_geom["position_mm"] = {"x": x_mm, "y": y_mm}

        # Convert position to pixels
        x_px = int(x_mm * scale_x)
        y_px = int(y_mm * scale_y)
        elem_geom["position_px"] = {"x": x_px, "y": y_px}

        # Size (if available)
        sz = elem.get("size", {})
        width_mm = sz.get("width_mm", 0)
        height_mm = sz.get("height_mm", 0)

        if width_mm > 0 and height_mm > 0:
            width_px = int(width_mm * scale_x)
            height_px = int(height_mm * scale_y)

            elem_geom["size_mm"] = {"width": width_mm, "height": height_mm}
            elem_geom["size_px"] = {"width": width_px, "height": height_px}

            elem_geom["bbox_mm"] = {
                "x0": x_mm,
                "y0": y_mm,
                "x1": x_mm + width_mm,
                "y1": y_mm + height_mm,
            }
            elem_geom["bbox_px"] = {
                "x": x_px,
                "y": y_px,
                "width": width_px,
                "height": height_px,
            }

        # Shape-specific geometry
        if elem.get("type") == "shape":
            start = elem.get("start", {})
            end = elem.get("end", {})
            if start and end:
                elem_geom["start_mm"] = start
                elem_geom["end_mm"] = end

                # Start/end in pixels
                start_x_px = int(start.get("x_mm", 0) * scale_x)
                start_y_px = int(start.get("y_mm", 0) * scale_y)
                end_x_px = int(end.get("x_mm", 0) * scale_x)
                end_y_px = int(end.get("y_mm", 0) * scale_y)
                elem_geom["start_px"] = {"x": start_x_px, "y": start_y_px}
                elem_geom["end_px"] = {"x": end_x_px, "y": end_y_px}

                # Calculate bounding box for shapes
                x0_mm = min(start.get("x_mm", 0), end.get("x_mm", 0))
                y0_mm = min(start.get("y_mm", 0), end.get("y_mm", 0))
                x1_mm = max(start.get("x_mm", 0), end.get("x_mm", 0))
                y1_mm = max(start.get("y_mm", 0), end.get("y_mm", 0))
                elem_geom["bbox_mm"] = {
                    "x0": x0_mm,
                    "y0": y0_mm,
                    "x1": x1_mm,
                    "y1": y1_mm,
                }

                x0_px = int(x0_mm * scale_x)
                y0_px = int(y0_mm * scale_y)
                w_px = int((x1_mm - x0_mm) * scale_x)
                h_px = int((y1_mm - y0_mm) * scale_y)
                elem_geom["bbox_px"] = {
                    "x": x0_px,
                    "y": y0_px,
                    "width": w_px,
                    "height": h_px,
                }

        # Text/symbol/equation elements - compute hit box from position
        if elem.get("type") in ("text", "symbol", "equation", "comment"):
            # Use a default hit box size for point elements
            hit_size_mm = 10 if elem.get("type") == "comment" else 5
            hit_size_px = int(hit_size_mm * scale_x)
            elem_geom["bbox_px"] = {
                "x": max(0, x_px - hit_size_px // 2),
                "y": max(0, y_px - hit_size_px // 2),
                "width": hit_size_px,
                "height": hit_size_px,
            }

        geometry["elements"].append(elem_geom)

    return geometry


def get_element_color_map(
    elements: List[Dict[str, Any]],
) -> Dict[str, Tuple[int, int, int]]:
    """Generate unique colors for each element for hitmap rendering.

    Args:
        elements: List of element specifications

    Returns:
        Dict mapping element_id to RGB color tuple
    """
    color_map = {}

    # Reserve (0, 0, 0) for background
    # Use distinct colors for each element
    for idx, elem in enumerate(elements):
        elem_id = elem.get("id", f"element_{idx}")
        # Generate unique color: use element index to create distinct RGB values
        # Avoid (0,0,0) which is background
        r = ((idx + 1) * 73) % 255 + 1  # +1 to avoid 0
        g = ((idx + 1) * 127) % 255
        b = ((idx + 1) * 199) % 255
        color_map[elem_id] = (r, g, b)

    return color_map


def color_to_element_id(
    color: Tuple[int, int, int],
    color_map: Dict[str, Tuple[int, int, int]],
    tolerance: int = 5,
) -> Optional[str]:
    """Find element ID from a color value.

    Args:
        color: RGB color tuple
        color_map: Dict mapping element_id to RGB color
        tolerance: Color matching tolerance

    Returns:
        Element ID if found, None otherwise
    """
    if sum(color[:3]) < 10:  # Near black = background
        return None

    for elem_id, elem_color in color_map.items():
        if all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color[:3], elem_color)):
            return elem_id

    return None


# EOF
