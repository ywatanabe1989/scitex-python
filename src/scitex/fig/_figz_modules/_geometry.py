#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_geometry.py

"""Geometry extraction for Figz bundles (GUI editing support)."""

from typing import Any, Dict, List


def extract_geometry(elements: List[Dict[str, Any]], size_mm: Dict[str, float]) -> dict:
    """Extract geometry data for all elements (hit areas for GUI editing).

    Args:
        elements: List of element specifications
        size_mm: Canvas size {"width": mm, "height": mm}

    Returns:
        Geometry data with element positions, sizes, and hit areas
    """
    geometry = {
        "canvas": {
            "width_mm": size_mm.get("width", 170),
            "height_mm": size_mm.get("height", 120),
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

        # Size (if available)
        sz = elem.get("size", {})
        width_mm = sz.get("width_mm", 0)
        height_mm = sz.get("height_mm", 0)

        if width_mm > 0 and height_mm > 0:
            elem_geom["size_mm"] = {"width": width_mm, "height": height_mm}
            elem_geom["bbox_mm"] = {
                "x0": x_mm,
                "y0": y_mm,
                "x1": x_mm + width_mm,
                "y1": y_mm + height_mm,
            }

        # Shape-specific geometry
        if elem.get("type") == "shape":
            start = elem.get("start", {})
            end = elem.get("end", {})
            if start and end:
                elem_geom["start_mm"] = start
                elem_geom["end_mm"] = end
                # Calculate bounding box for shapes
                x0 = min(start.get("x_mm", 0), end.get("x_mm", 0))
                y0 = min(start.get("y_mm", 0), end.get("y_mm", 0))
                x1 = max(start.get("x_mm", 0), end.get("x_mm", 0))
                y1 = max(start.get("y_mm", 0), end.get("y_mm", 0))
                elem_geom["bbox_mm"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

        geometry["elements"].append(elem_geom)

    return geometry


# EOF
