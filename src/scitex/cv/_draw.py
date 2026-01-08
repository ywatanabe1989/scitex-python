#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/cv/_draw.py
"""Drawing utilities using cv2."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def rectangle(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    filled: bool = False,
) -> np.ndarray:
    """Draw a rectangle on an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (modified in place).
    pt1 : tuple
        Top-left corner (x, y).
    pt2 : tuple
        Bottom-right corner (x, y).
    color : tuple
        BGR color.
    thickness : int
        Line thickness (-1 for filled).
    filled : bool
        If True, fill the rectangle.

    Returns
    -------
    np.ndarray
        Image with rectangle.
    """
    t = -1 if filled else thickness
    cv2.rectangle(img, pt1, pt2, color, t)
    return img


def circle(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    filled: bool = False,
) -> np.ndarray:
    """Draw a circle on an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (modified in place).
    center : tuple
        Center coordinates (x, y).
    radius : int
        Circle radius.
    color : tuple
        BGR color.
    thickness : int
        Line thickness.
    filled : bool
        If True, fill the circle.

    Returns
    -------
    np.ndarray
        Image with circle.
    """
    t = -1 if filled else thickness
    cv2.circle(img, center, radius, color, t)
    return img


def line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a line on an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (modified in place).
    pt1 : tuple
        Start point (x, y).
    pt2 : tuple
        End point (x, y).
    color : tuple
        BGR color.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Image with line.
    """
    cv2.line(img, pt1, pt2, color, thickness)
    return img


def text(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 1.0,
    thickness: int = 2,
    font: str = "simplex",
) -> np.ndarray:
    """Draw text on an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (modified in place).
    text : str
        Text to draw.
    position : tuple
        Bottom-left corner of text (x, y).
    color : tuple
        BGR color.
    scale : float
        Font scale.
    thickness : int
        Text thickness.
    font : str
        Font type: 'simplex', 'plain', 'duplex', 'complex', 'triplex'.

    Returns
    -------
    np.ndarray
        Image with text.
    """
    font_map = {
        "simplex": cv2.FONT_HERSHEY_SIMPLEX,
        "plain": cv2.FONT_HERSHEY_PLAIN,
        "duplex": cv2.FONT_HERSHEY_DUPLEX,
        "complex": cv2.FONT_HERSHEY_COMPLEX,
        "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    }
    font_face = font_map.get(font, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.putText(img, text, position, font_face, scale, color, thickness)
    return img


def polylines(
    img: np.ndarray,
    points: np.ndarray,
    closed: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw polylines on an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (modified in place).
    points : np.ndarray
        Array of points with shape (N, 1, 2) or (N, 2).
    closed : bool
        Whether to close the polyline.
    color : tuple
        BGR color.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Image with polylines.
    """
    if len(points.shape) == 2:
        points = points.reshape((-1, 1, 2))
    cv2.polylines(img, [points.astype(np.int32)], closed, color, thickness)
    return img


def arrow(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    tip_length: float = 0.1,
) -> np.ndarray:
    """Draw an arrowed line on an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (modified in place).
    pt1 : tuple
        Start point (x, y).
    pt2 : tuple
        End point (arrow tip) (x, y).
    color : tuple
        BGR color.
    thickness : int
        Line thickness.
    tip_length : float
        Arrow tip length as fraction of line length.

    Returns
    -------
    np.ndarray
        Image with arrow.
    """
    cv2.arrowedLine(img, pt1, pt2, color, thickness, tipLength=tip_length)
    return img


__all__ = [
    "rectangle",
    "circle",
    "line",
    "text",
    "polylines",
    "arrow",
]

# EOF
