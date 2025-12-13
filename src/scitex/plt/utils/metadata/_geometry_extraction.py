#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: scitex/plt/utils/metadata/_geometry_extraction.py

"""
Geometry extraction utilities for Schema v0.3.

This module provides functions to extract shape geometry in axes-local pixel
coordinates from matplotlib artists for interactive hit-testing in /vis/.

All coordinates are in AXES-LOCAL pixels:
- Origin at axes bounding box top-left
- X increases right, Y increases down (screen coordinates)

Geometry types per artist:
- line: path_simplified (polyline)
- scatter: points with hit_radius_px
- fill_between/polygon: polygon (closed shape)
- bar/rectangle: rectangles array
- image/contour: bbox only
- text: bbox + anchor
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def extract_axes_bbox_px(ax, fig) -> Dict[str, int]:
    """
    Extract axes bounding box in figure pixel coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object

    Returns
    -------
    dict
        Bounding box with keys: x0, y0, x1, y1 (figure pixels)
    """
    # Get axes position in figure coordinates (0-1)
    bbox = ax.get_position()

    # Get figure size in pixels
    fig_width_px = fig.get_figwidth() * fig.dpi
    fig_height_px = fig.get_figheight() * fig.dpi

    # Convert to figure pixels
    x0 = int(bbox.x0 * fig_width_px)
    y0 = int((1 - bbox.y1) * fig_height_px)  # Flip Y for screen coords
    x1 = int(bbox.x1 * fig_width_px)
    y1 = int((1 - bbox.y0) * fig_height_px)  # Flip Y for screen coords

    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def data_to_axes_px(ax, fig, x_data, y_data) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert data coordinates to axes-local pixel coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object
    x_data : array-like
        X coordinates in data units
    y_data : array-like
        Y coordinates in data units

    Returns
    -------
    tuple
        (x_px, y_px) in axes-local pixels
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Get axes bbox in figure pixels
    axes_bbox = extract_axes_bbox_px(ax, fig)
    ax_width = axes_bbox["x1"] - axes_bbox["x0"]
    ax_height = axes_bbox["y1"] - axes_bbox["y0"]

    # Get data limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Convert data to normalized axes coordinates (0-1)
    x_norm = (x_data - xlim[0]) / (xlim[1] - xlim[0])
    y_norm = (y_data - ylim[0]) / (ylim[1] - ylim[0])

    # Convert to axes-local pixels (Y flipped for screen coords)
    x_px = x_norm * ax_width
    y_px = (1 - y_norm) * ax_height  # Flip Y

    return x_px.astype(float), y_px.astype(float)


def extract_line_geometry(line, ax, fig, simplify_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Extract line geometry in axes-local pixels.

    Parameters
    ----------
    line : matplotlib.lines.Line2D
        The line object
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object
    simplify_threshold : float
        Maximum error in pixels for path simplification (Douglas-Peucker)

    Returns
    -------
    dict
        Geometry with: coord_space, bbox, path_simplified, (optional) path
    """
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    if len(x_data) == 0:
        return {"coord_space": "axes", "bbox": None, "path_simplified": []}

    x_px, y_px = data_to_axes_px(ax, fig, x_data, y_data)

    # Build path as list of [x, y] points
    path = [[round(x, 1), round(y, 1)] for x, y in zip(x_px, y_px)]

    # Simplify path using Douglas-Peucker
    path_simplified = _simplify_path(path, simplify_threshold)

    # Compute bounding box
    bbox = _compute_bbox(x_px, y_px)

    return {
        "coord_space": "axes",
        "bbox": bbox,
        "path_simplified": path_simplified,
        "path": None  # Full path omitted for performance
    }


def extract_scatter_geometry(collection, ax, fig) -> Dict[str, Any]:
    """
    Extract scatter point geometry in axes-local pixels.

    Parameters
    ----------
    collection : matplotlib.collections.PathCollection
        The scatter collection
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object

    Returns
    -------
    dict
        Geometry with: coord_space, bbox, hit_radius_px, points
    """
    offsets = collection.get_offsets()

    if len(offsets) == 0:
        return {"coord_space": "axes", "bbox": None, "hit_radius_px": 6.0, "points": []}

    x_data = offsets[:, 0]
    y_data = offsets[:, 1]

    x_px, y_px = data_to_axes_px(ax, fig, x_data, y_data)

    # Get marker sizes (s is area in points^2)
    sizes = collection.get_sizes()
    if len(sizes) == 0:
        sizes = [36]  # Default matplotlib marker size

    # Compute hit radius from average marker size
    # s = area in points^2, radius = sqrt(s/pi) in points
    # Convert points to pixels (roughly 1:1 at 72 dpi, scale with actual dpi)
    avg_size = np.mean(sizes)
    radius_pt = np.sqrt(avg_size / np.pi)
    hit_radius_px = max(radius_pt * fig.dpi / 72, 5.0)  # Min 5px for usability

    # Build points list
    points = [{"x": round(x, 1), "y": round(y, 1)} for x, y in zip(x_px, y_px)]

    # Compute bounding box
    bbox = _compute_bbox(x_px, y_px)

    return {
        "coord_space": "axes",
        "bbox": bbox,
        "hit_radius_px": round(hit_radius_px, 1),
        "points": points
    }


def extract_polygon_geometry(collection, ax, fig) -> Dict[str, Any]:
    """
    Extract polygon geometry (fill_between, violin, etc.) in axes-local pixels.

    Parameters
    ----------
    collection : matplotlib.collections.PolyCollection
        The polygon collection
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object

    Returns
    -------
    dict
        Geometry with: coord_space, bbox, polygon
    """
    paths = collection.get_paths()

    if len(paths) == 0:
        return {"coord_space": "axes", "bbox": None, "polygon": []}

    # Get vertices from first path
    vertices = paths[0].vertices

    if len(vertices) == 0:
        return {"coord_space": "axes", "bbox": None, "polygon": []}

    x_data = vertices[:, 0]
    y_data = vertices[:, 1]

    x_px, y_px = data_to_axes_px(ax, fig, x_data, y_data)

    # Build polygon as list of [x, y] points
    polygon = [[round(x, 1), round(y, 1)] for x, y in zip(x_px, y_px)]

    # Simplify if too many points
    if len(polygon) > 100:
        polygon = _simplify_path(polygon, 1.0)

    # Compute bounding box
    bbox = _compute_bbox(x_px, y_px)

    return {
        "coord_space": "axes",
        "bbox": bbox,
        "polygon": polygon
    }


def extract_rectangle_geometry(patch, ax, fig) -> Dict[str, Any]:
    """
    Extract rectangle geometry (bar, etc.) in axes-local pixels.

    Parameters
    ----------
    patch : matplotlib.patches.Rectangle
        The rectangle patch
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object

    Returns
    -------
    dict
        Geometry with: coord_space, bbox (also serves as the rectangle)
    """
    # Get rectangle corners in data coordinates
    x0_data = patch.get_x()
    y0_data = patch.get_y()
    width_data = patch.get_width()
    height_data = patch.get_height()

    # Convert corners to axes-local pixels
    x_corners = [x0_data, x0_data + width_data]
    y_corners = [y0_data, y0_data + height_data]

    x_px, _ = data_to_axes_px(ax, fig, x_corners, [y0_data, y0_data])
    _, y_px = data_to_axes_px(ax, fig, [x0_data, x0_data], y_corners)

    # Build rectangle (note: Y may be flipped)
    x = round(min(x_px), 1)
    y = round(min(y_px), 1)
    width = round(abs(x_px[1] - x_px[0]), 1)
    height = round(abs(y_px[1] - y_px[0]), 1)

    return {
        "coord_space": "axes",
        "bbox": {"x0": int(x), "y0": int(y), "x1": int(x + width), "y1": int(y + height)},
        "rectangle": {"x": x, "y": y, "width": width, "height": height}
    }


def extract_bar_group_geometry(patches, ax, fig) -> Dict[str, Any]:
    """
    Extract geometry for a group of bar rectangles.

    Parameters
    ----------
    patches : list
        List of Rectangle patches
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object

    Returns
    -------
    dict
        Geometry with: coord_space, bbox, rectangles
    """
    rectangles = []
    all_x = []
    all_y = []

    for patch in patches:
        geom = extract_rectangle_geometry(patch, ax, fig)
        rect = geom.get("rectangle", {})
        if rect:
            rectangles.append(rect)
            all_x.extend([rect["x"], rect["x"] + rect["width"]])
            all_y.extend([rect["y"], rect["y"] + rect["height"]])

    if not rectangles:
        return {"coord_space": "axes", "bbox": None, "rectangles": []}

    # Compute overall bounding box
    bbox = {
        "x0": int(min(all_x)),
        "y0": int(min(all_y)),
        "x1": int(max(all_x)),
        "y1": int(max(all_y))
    }

    return {
        "coord_space": "axes",
        "bbox": bbox,
        "rectangles": rectangles
    }


def extract_text_geometry(text, ax, fig) -> Dict[str, Any]:
    """
    Extract text geometry in axes-local pixels.

    Parameters
    ----------
    text : matplotlib.text.Text
        The text object
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object

    Returns
    -------
    dict
        Geometry with: coord_space, bbox, anchor
    """
    # Get text position in data coordinates
    x_data, y_data = text.get_position()

    # Convert to axes-local pixels
    x_px, y_px = data_to_axes_px(ax, fig, [x_data], [y_data])

    anchor = {"x": float(round(x_px[0], 1)), "y": float(round(y_px[0], 1))}

    # Try to get text bounding box
    try:
        renderer = fig.canvas.get_renderer()
        bbox_display = text.get_window_extent(renderer=renderer)

        # Get axes bbox for conversion
        axes_bbox = extract_axes_bbox_px(ax, fig)

        # Convert display coords to axes-local pixels
        x0 = bbox_display.x0 - axes_bbox["x0"]
        y0 = (fig.get_figheight() * fig.dpi - bbox_display.y1) - axes_bbox["y0"]  # Flip Y
        x1 = bbox_display.x1 - axes_bbox["x0"]
        y1 = (fig.get_figheight() * fig.dpi - bbox_display.y0) - axes_bbox["y0"]

        bbox = {
            "x0": int(round(x0)),
            "y0": int(round(y0)),
            "x1": int(round(x1)),
            "y1": int(round(y1))
        }
    except Exception:
        # Fallback: estimate bbox from anchor
        bbox = {
            "x0": int(anchor["x"] - 20),
            "y0": int(anchor["y"] - 10),
            "x1": int(anchor["x"] + 80),
            "y1": int(anchor["y"] + 10)
        }

    return {
        "coord_space": "axes",
        "bbox": bbox,
        "anchor": anchor
    }


def extract_image_geometry(image, ax, fig) -> Dict[str, Any]:
    """
    Extract image geometry (imshow, etc.) in axes-local pixels.

    Parameters
    ----------
    image : matplotlib.image.AxesImage
        The image object
    ax : matplotlib.axes.Axes
        The axes object
    fig : matplotlib.figure.Figure
        The figure object

    Returns
    -------
    dict
        Geometry with: coord_space, bbox
    """
    extent = image.get_extent()

    if extent is None:
        # Full axes
        return {
            "coord_space": "axes",
            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 100}  # Will be computed
        }

    x0_data, x1_data, y0_data, y1_data = extent

    x_px, _ = data_to_axes_px(ax, fig, [x0_data, x1_data], [y0_data, y0_data])
    _, y_px = data_to_axes_px(ax, fig, [x0_data, x0_data], [y0_data, y1_data])

    bbox = {
        "x0": int(round(min(x_px))),
        "y0": int(round(min(y_px))),
        "x1": int(round(max(x_px))),
        "y1": int(round(max(y_px)))
    }

    return {
        "coord_space": "axes",
        "bbox": bbox
    }


# =============================================================================
# Helper functions
# =============================================================================

def _compute_bbox(x_px: np.ndarray, y_px: np.ndarray) -> Dict[str, int]:
    """Compute bounding box from pixel coordinates."""
    return {
        "x0": int(round(np.nanmin(x_px))),
        "y0": int(round(np.nanmin(y_px))),
        "x1": int(round(np.nanmax(x_px))),
        "y1": int(round(np.nanmax(y_px)))
    }


def _simplify_path(path: List[List[float]], tolerance: float) -> List[List[float]]:
    """
    Simplify a path using Douglas-Peucker algorithm.

    Parameters
    ----------
    path : list
        List of [x, y] points
    tolerance : float
        Maximum perpendicular distance in pixels

    Returns
    -------
    list
        Simplified path
    """
    if len(path) <= 2:
        return path

    # Convert to numpy for easier computation
    points = np.array(path)

    # Find point with maximum distance from line between endpoints
    start = points[0]
    end = points[-1]

    # Line vector
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    # Handle closed polygons (start ≈ end)
    if line_len < 1e-10:
        # Find point farthest from centroid to split the polygon
        centroid = np.mean(points, axis=0)
        distances_from_centroid = np.linalg.norm(points - centroid, axis=1)
        split_idx = np.argmax(distances_from_centroid)

        if split_idx == 0 or split_idx == len(points) - 1:
            # Can't split well, return simplified version
            return _simplify_open_path(path, tolerance)

        # Split and simplify each half
        first_half = [p.tolist() for p in points[:split_idx + 1]]
        second_half = [p.tolist() for p in points[split_idx:]]

        simplified_first = _simplify_open_path(first_half, tolerance)
        simplified_second = _simplify_open_path(second_half, tolerance)

        # Join halves (remove duplicate at split point)
        return simplified_first[:-1] + simplified_second

    return _simplify_open_path(path, tolerance)


def _simplify_open_path(path: List[List[float]], tolerance: float) -> List[List[float]]:
    """
    Simplify an open path using Douglas-Peucker algorithm.

    Parameters
    ----------
    path : list
        List of [x, y] points (open path, start != end)
    tolerance : float
        Maximum perpendicular distance in pixels

    Returns
    -------
    list
        Simplified path
    """
    if len(path) <= 2:
        return path

    points = np.array(path)
    start = points[0]
    end = points[-1]

    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return [path[0], path[-1]]

    line_unit = line_vec / line_len

    # Perpendicular distances
    point_vecs = points - start
    projections = np.dot(point_vecs, line_unit)
    closest_points = start + np.outer(projections, line_unit)
    distances = np.linalg.norm(points - closest_points, axis=1)

    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]

    if max_dist > tolerance:
        # Recurse
        left = _simplify_open_path([p.tolist() for p in points[:max_idx + 1]], tolerance)
        right = _simplify_open_path([p.tolist() for p in points[max_idx:]], tolerance)
        return left[:-1] + right
    else:
        return [path[0], path[-1]]


# EOF
