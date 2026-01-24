#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap/_color_conversion.py

"""
Color conversion functions for hitmap ID encoding.

This module provides functions to convert between element IDs and RGB colors
for pixel-perfect element identification in hit maps.
"""

import colorsys
import hashlib
from typing import Tuple

__all__ = [
    "id_to_rgb",
    "rgb_to_id",
    "rgb_to_id_lookup",
]

# Hand-picked palette for first 12 elements (most common case)
DISTINCT_COLORS = [
    (255, 0, 0),  # 1: Red
    (0, 200, 0),  # 2: Green
    (0, 100, 255),  # 3: Blue
    (255, 200, 0),  # 4: Yellow/Gold
    (255, 0, 200),  # 5: Magenta/Pink
    (0, 220, 220),  # 6: Cyan
    (255, 100, 0),  # 7: Orange
    (150, 0, 255),  # 8: Purple
    (0, 255, 100),  # 9: Spring Green
    (255, 100, 150),  # 10: Salmon/Rose
    (100, 255, 0),  # 11: Lime
    (100, 150, 255),  # 12: Sky Blue
]


def id_to_rgb(element_id: int) -> Tuple[int, int, int]:
    """
    Convert element ID to unique, human-readable RGB color.

    Uses a hash function to generate visually distinct colors that are:
    1. Deterministic (same ID always gives same color)
    2. Visually distinct (spread across the color space)
    3. Bright and saturated (easy to see)

    The first 12 elements use a hand-picked palette for maximum distinctness.
    Beyond that, uses hash-based HSV generation with high saturation.

    Parameters
    ----------
    element_id : int
        Element ID (1-based). ID 0 is reserved for background.

    Returns
    -------
    tuple
        (R, G, B) values (0-255)
    """
    if element_id <= 0:
        return (0, 0, 0)  # Background

    if element_id <= len(DISTINCT_COLORS):
        return DISTINCT_COLORS[element_id - 1]

    # For IDs > 12, use hash-based color generation
    hash_bytes = hashlib.md5(str(element_id).encode()).digest()

    # Use hash bytes to generate HSV values
    hue = int.from_bytes(hash_bytes[0:2], "big") / 65535.0
    saturation = 0.7 + (int.from_bytes(hash_bytes[2:3], "big") / 255.0) * 0.3
    value = 0.75 + (int.from_bytes(hash_bytes[3:4], "big") / 255.0) * 0.25

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_id(r: int, g: int, b: int) -> int:
    """
    Convert RGB color back to element ID using 24-bit encoding.

    Parameters
    ----------
    r, g, b : int
        RGB values (0-255)

    Returns
    -------
    int
        Element ID
    """
    return (r << 16) | (g << 8) | b


def rgb_to_id_lookup(r: int, g: int, b: int, color_map: dict) -> int:
    """
    Convert RGB color back to element ID using the color map.

    Since we use human-readable colors, we need to look up in the map.

    Parameters
    ----------
    r, g, b : int
        RGB values (0-255)
    color_map : dict
        Color map from generate_hitmap_id_colors (maps ID -> info with 'rgb' key)

    Returns
    -------
    int
        Element ID, or 0 if not found
    """
    rgb = [r, g, b]
    for element_id, info in color_map.items():
        if info.get("rgb") == rgb:
            return element_id
    return 0


# EOF
