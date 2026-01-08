#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/cv/_transform.py
"""Image transformation utilities using cv2."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np


def resize(
    img: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    interpolation: str = "linear",
) -> np.ndarray:
    """Resize an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    size : tuple of int, optional
        Target size as (width, height).
    scale : float, optional
        Scale factor (alternative to size).
    interpolation : str
        Interpolation method: 'nearest', 'linear', 'cubic', 'area', 'lanczos'.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

    if scale is not None:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    elif size is not None:
        return cv2.resize(img, size, interpolation=interp)
    else:
        raise ValueError("Either size or scale must be provided")


def rotate(
    img: np.ndarray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Rotate an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    angle : float
        Rotation angle in degrees (counter-clockwise).
    center : tuple of int, optional
        Rotation center. Defaults to image center.
    scale : float
        Scale factor.

    Returns
    -------
    np.ndarray
        Rotated image.
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, matrix, (w, h))


def flip(
    img: np.ndarray,
    direction: str = "horizontal",
) -> np.ndarray:
    """Flip an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    direction : str
        Flip direction: 'horizontal', 'vertical', or 'both'.

    Returns
    -------
    np.ndarray
        Flipped image.
    """
    flip_map = {
        "horizontal": 1,
        "vertical": 0,
        "both": -1,
    }
    code = flip_map.get(direction, 1)
    return cv2.flip(img, code)


def crop(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Crop a region from an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    x, y : int
        Top-left corner coordinates.
    width, height : int
        Crop dimensions.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    return img[y : y + height, x : x + width].copy()


def pad(
    img: np.ndarray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    color: Union[int, Tuple[int, ...]] = 0,
    mode: str = "constant",
) -> np.ndarray:
    """Pad an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    top, bottom, left, right : int
        Padding sizes.
    color : int or tuple
        Padding color for constant mode.
    mode : str
        Padding mode: 'constant', 'reflect', 'replicate'.

    Returns
    -------
    np.ndarray
        Padded image.
    """
    mode_map = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }
    border_type = mode_map.get(mode, cv2.BORDER_CONSTANT)
    return cv2.copyMakeBorder(img, top, bottom, left, right, border_type, value=color)


__all__ = [
    "resize",
    "rotate",
    "flip",
    "crop",
    "pad",
]

# EOF
