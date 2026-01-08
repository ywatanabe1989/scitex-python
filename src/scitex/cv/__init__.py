#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/cv/__init__.py
"""SciTeX Computer Vision Module.

Provides reusable cv2-based utilities for image processing:
- I/O: load, save, color conversions
- Transform: resize, rotate, flip, crop, pad
- Filters: blur, sharpen, edge detection, threshold, denoise
- Draw: rectangle, circle, line, text, polylines, arrow

Example
-------
>>> import scitex.cv as cv
>>> # Load and process an image
>>> img = cv.load("input.png")
>>> img = cv.resize(img, scale=0.5)
>>> img = cv.blur(img, ksize=5)
>>> edges = cv.edge_detect(img, method="canny")
>>> cv.save(edges, "edges.png")
"""

# I/O
# Drawing
from ._draw import (
    arrow,
    circle,
    line,
    polylines,
    rectangle,
    text,
)

# Filters
from ._filters import (
    blur,
    denoise,
    edge_detect,
    sharpen,
    threshold,
)
from ._io import (
    load,
    save,
    to_bgr,
    to_gray,
    to_rgb,
)

# Transforms
from ._transform import (
    crop,
    flip,
    pad,
    resize,
    rotate,
)

__all__ = [
    # I/O
    "load",
    "save",
    "to_rgb",
    "to_bgr",
    "to_gray",
    # Transforms
    "resize",
    "rotate",
    "flip",
    "crop",
    "pad",
    # Filters
    "blur",
    "sharpen",
    "edge_detect",
    "threshold",
    "denoise",
    # Drawing
    "rectangle",
    "circle",
    "line",
    "text",
    "polylines",
    "arrow",
]

# EOF
