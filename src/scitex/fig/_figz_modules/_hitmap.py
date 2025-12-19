#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_hitmap.py

"""Hitmap generation for Figz bundles (pixel-level hit detection)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def generate_hitmap(
    elements: List[Dict[str, Any]],
    size_mm: Dict[str, float],
    cache_dir: Path,
    dpi: int = 150,
) -> Tuple[Optional[int], Optional[int], Optional[Dict[str, Tuple[int, int, int]]]]:
    """Generate hitmap image for GUI hit testing.

    Creates a color-coded image where each element has a unique solid color,
    enabling pixel-perfect hit detection by color lookup.

    Args:
        elements: List of element specifications
        size_mm: Canvas size {"width": mm, "height": mm}
        cache_dir: Cache directory for output files
        dpi: Resolution for rendering

    Returns:
        Tuple of (actual_width_px, actual_height_px, color_map)
    """
    from PIL import Image, ImageDraw

    from ._geometry import get_element_color_map

    try:
        # Calculate canvas size in pixels
        width_mm = size_mm.get("width", 170)
        height_mm = size_mm.get("height", 120)
        width_px = int(width_mm * dpi / 25.4)
        height_px = int(height_mm * dpi / 25.4)

        # Generate unique colors for each element
        color_map = get_element_color_map(elements)

        # Create hitmap image with black background (background = no element)
        hitmap = Image.new("RGB", (width_px, height_px), (0, 0, 0))
        draw = ImageDraw.Draw(hitmap)

        # Calculate scale factors
        scale_x = dpi / 25.4
        scale_y = dpi / 25.4

        # Draw each element as a solid colored rectangle
        for elem in elements:
            elem_id = elem.get("id")
            if not elem_id or elem_id not in color_map:
                continue

            color = color_map[elem_id]
            _draw_element_hitbox(draw, elem, color, scale_x, scale_y)

        # Save hitmap
        hitmap.save(cache_dir / "hitmap.png")

        # Save color map for lookup
        color_map_data = {
            elem_id: {"r": c[0], "g": c[1], "b": c[2]}
            for elem_id, c in color_map.items()
        }
        with open(cache_dir / "hitmap_colors.json", "w") as f:
            json.dump(color_map_data, f, indent=2)

        # Update or create render_manifest.json
        _update_manifest(cache_dir, width_px, height_px)

        return (width_px, height_px, color_map)

    except Exception:
        return (None, None, None)


def _draw_element_hitbox(
    draw,
    elem: Dict[str, Any],
    color: Tuple[int, int, int],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw a hitbox for an element on the hitmap.

    Args:
        draw: PIL ImageDraw instance
        elem: Element specification
        color: RGB color tuple for this element
        scale_x: X scale factor (mm to px)
        scale_y: Y scale factor (mm to px)
    """
    elem_type = elem.get("type")

    if elem_type in ("plot", "image", "figure"):
        _draw_rect_element(draw, elem, color, scale_x, scale_y)
    elif elem_type == "shape":
        _draw_shape_element(draw, elem, color, scale_x, scale_y)
    elif elem_type in ("text", "symbol", "equation", "comment"):
        _draw_point_element(draw, elem, color, scale_x, scale_y)


def _draw_rect_element(
    draw,
    elem: Dict[str, Any],
    color: Tuple[int, int, int],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw rectangular element (plot, image, figure)."""
    pos = elem.get("position", {})
    sz = elem.get("size", {})
    x_mm = pos.get("x_mm", 0)
    y_mm = pos.get("y_mm", 0)
    w_mm = sz.get("width_mm", 0)
    h_mm = sz.get("height_mm", 0)

    if w_mm > 0 and h_mm > 0:
        x0 = int(x_mm * scale_x)
        y0 = int(y_mm * scale_y)
        x1 = int((x_mm + w_mm) * scale_x)
        y1 = int((y_mm + h_mm) * scale_y)
        draw.rectangle([x0, y0, x1, y1], fill=color)


def _draw_shape_element(
    draw,
    elem: Dict[str, Any],
    color: Tuple[int, int, int],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw shape element (arrow, line, bracket)."""
    start = elem.get("start", {})
    end = elem.get("end", {})
    if start and end:
        x0_mm = min(start.get("x_mm", 0), end.get("x_mm", 0))
        y0_mm = min(start.get("y_mm", 0), end.get("y_mm", 0))
        x1_mm = max(start.get("x_mm", 0), end.get("x_mm", 0))
        y1_mm = max(start.get("y_mm", 0), end.get("y_mm", 0))

        # Add padding for thin shapes (lines, arrows)
        padding_mm = 2
        x0 = int((x0_mm - padding_mm) * scale_x)
        y0 = int((y0_mm - padding_mm) * scale_y)
        x1 = int((x1_mm + padding_mm) * scale_x)
        y1 = int((y1_mm + padding_mm) * scale_y)
        draw.rectangle([x0, y0, x1, y1], fill=color)


def _draw_point_element(
    draw,
    elem: Dict[str, Any],
    color: Tuple[int, int, int],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw point element (text, symbol, equation, comment)."""
    pos = elem.get("position", {})
    x_mm = pos.get("x_mm", 0)
    y_mm = pos.get("y_mm", 0)
    elem_type = elem.get("type")

    # Default hit box size
    hit_size_mm = 10 if elem_type == "comment" else 5
    half_size = int(hit_size_mm * scale_x / 2)
    cx = int(x_mm * scale_x)
    cy = int(y_mm * scale_y)

    draw.rectangle(
        [cx - half_size, cy - half_size, cx + half_size, cy + half_size],
        fill=color,
    )


def _update_manifest(cache_dir: Path, width_px: int, height_px: int) -> None:
    """Update render_manifest.json with hitmap info."""
    manifest_path = cache_dir / "render_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest["hitmap_png"] = "cache/hitmap.png"
    manifest["hitmap_colors"] = "cache/hitmap_colors.json"
    manifest["hitmap_size_px"] = {"width": width_px, "height": height_px}

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def hit_test_pixel(
    x_px: int,
    y_px: int,
    hitmap_path: Path,
    color_map_path: Path,
) -> Optional[str]:
    """Perform hit test at pixel coordinates using hitmap.

    Args:
        x_px: X coordinate in pixels
        y_px: Y coordinate in pixels
        hitmap_path: Path to hitmap.png
        color_map_path: Path to hitmap_colors.json

    Returns:
        Element ID at the pixel, or None if background
    """
    from PIL import Image

    try:
        hitmap = Image.open(hitmap_path)
        with open(color_map_path) as f:
            color_map_data = json.load(f)

        # Get pixel color
        if 0 <= x_px < hitmap.width and 0 <= y_px < hitmap.height:
            pixel = hitmap.getpixel((x_px, y_px))
            r, g, b = pixel[:3]

            # Background check (black)
            if r < 10 and g < 10 and b < 10:
                return None

            # Find matching element
            for elem_id, color in color_map_data.items():
                if (
                    abs(color["r"] - r) <= 5
                    and abs(color["g"] - g) <= 5
                    and abs(color["b"] - b) <= 5
                ):
                    return elem_id

        return None
    except Exception:
        return None


# EOF
