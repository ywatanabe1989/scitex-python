#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap/_query.py

"""
Hitmap query and save functions.

This module provides functions to query hitmap data and save hitmaps as PNG files.
"""

from typing import Any, Dict, List

import numpy as np

__all__ = [
    "query_hitmap_neighborhood",
    "save_hitmap_png",
]


def query_hitmap_neighborhood(
    hitmap: np.ndarray,
    x: int,
    y: int,
    color_map: Dict[int, Dict[str, Any]],
    radius: int = 2,
) -> List[Dict[str, Any]]:
    """
    Query hit map with neighborhood sampling for smart selection.

    Finds all element IDs in a neighborhood around the click point,
    enabling selection of overlapping elements and thin lines.

    Parameters
    ----------
    hitmap : np.ndarray
        Hit map array (uint32, element IDs from RGB encoding).
    x : int
        X coordinate (column) of click point.
    y : int
        Y coordinate (row) of click point.
    color_map : dict
        Mapping from element ID to element info.
    radius : int
        Sampling radius (e.g., 2 = 5x5 neighborhood).

    Returns
    -------
    list of dict
        List of element info dicts for all elements found in neighborhood,
        sorted by distance from click point (closest first).

    Notes
    -----
    Use cases:
    - Alt+Click to select objects underneath (lower z-order)
    - Click on thin lines that might be missed with exact pixel
    - Show candidate list when multiple elements overlap
    """
    h, w = hitmap.shape
    found_ids = set()
    id_distances = {}

    # Sample neighborhood
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                element_id = int(hitmap[ny, nx])
                if element_id > 0 and element_id in color_map:
                    found_ids.add(element_id)
                    # Track minimum distance for each ID
                    dist = abs(dx) + abs(dy)  # Manhattan distance
                    if (
                        element_id not in id_distances
                        or dist < id_distances[element_id]
                    ):
                        id_distances[element_id] = dist

    # Sort by distance (closest first), then by ID for stability
    sorted_ids = sorted(found_ids, key=lambda eid: (id_distances[eid], eid))

    return [color_map[eid] for eid in sorted_ids]


def save_hitmap_png(hitmap: np.ndarray, path: str, color_map: Dict = None):
    """
    Save hit map as a PNG file (RGB encoding for 24-bit IDs).

    Parameters
    ----------
    hitmap : np.ndarray
        Hit map array (uint32, element IDs from 24-bit RGB encoding).
    path : str
        Output path for PNG file.
    color_map : dict, optional
        Color map for visualization (unused, kept for API compatibility).
    """
    from PIL import Image

    # Convert 24-bit IDs back to RGB for PNG storage
    h, w = hitmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (hitmap >> 16) & 0xFF  # R
    rgb[:, :, 1] = (hitmap >> 8) & 0xFF  # G
    rgb[:, :, 2] = hitmap & 0xFF  # B

    # Save as RGB PNG (preserves exact ID values)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path)


# EOF
