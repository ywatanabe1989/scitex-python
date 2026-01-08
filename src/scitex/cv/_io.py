#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/cv/_io.py
"""Image I/O utilities using cv2."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load(
    path: Union[str, Path],
    color: bool = True,
    alpha: bool = False,
) -> np.ndarray:
    """Load an image from file.

    Parameters
    ----------
    path : str or Path
        Image file path.
    color : bool
        If True, load as color (BGR). If False, load as grayscale.
    alpha : bool
        If True, preserve alpha channel (BGRA).

    Returns
    -------
    np.ndarray
        Image array in BGR or grayscale format.
    """
    path = str(path)
    if alpha:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif color:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def save(
    img: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
) -> Path:
    """Save an image to file.

    Parameters
    ----------
    img : np.ndarray
        Image array (BGR or grayscale).
    path : str or Path
        Output file path.
    quality : int
        JPEG quality (0-100) or PNG compression (0-9).

    Returns
    -------
    Path
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        # PNG compression is 0-9, map quality to this
        compression = max(0, min(9, (100 - quality) // 10))
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    else:
        params = []

    success = cv2.imwrite(str(path), img, params)
    if not success:
        raise OSError(f"Failed to save image: {path}")
    return path


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB.

    Parameters
    ----------
    img : np.ndarray
        BGR image.

    Returns
    -------
    np.ndarray
        RGB image.
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR.

    Parameters
    ----------
    img : np.ndarray
        RGB image.

    Returns
    -------
    np.ndarray
        BGR image.
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale.

    Parameters
    ----------
    img : np.ndarray
        Color image (BGR or RGB).

    Returns
    -------
    np.ndarray
        Grayscale image.
    """
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


__all__ = [
    "load",
    "save",
    "to_rgb",
    "to_bgr",
    "to_gray",
]

# EOF
