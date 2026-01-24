#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_utils.py

"""
Utility functions for DearPyGui editor.

Provides helper functions like checkerboard pattern creation for transparency preview.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def create_checkerboard(width: int, height: int, square_size: int = 10) -> "Image":
    """Create a checkerboard pattern image for transparency preview.

    Parameters
    ----------
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    square_size : int
        Size of each checkerboard square (default: 10)

    Returns
    -------
    PIL.Image
        RGBA image with checkerboard pattern (light/dark gray)
    """
    import numpy as np
    from PIL import Image

    # Create checkerboard pattern
    light_gray = (220, 220, 220, 255)
    dark_gray = (180, 180, 180, 255)

    # Create array
    img_array = np.zeros((height, width, 4), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Determine which square we're in
            square_x = x // square_size
            square_y = y // square_size
            if (square_x + square_y) % 2 == 0:
                img_array[y, x] = light_gray
            else:
                img_array[y, x] = dark_gray

    return Image.fromarray(img_array, "RGBA")


# mm to pt conversion factor
MM_TO_PT = 2.83465


# EOF
