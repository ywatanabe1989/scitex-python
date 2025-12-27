#!/usr/bin/env python3
# Timestamp: "2025-12-27 (refactored)"
# File: scitex/stats/descriptive/_circular.py
"""
Circular statistics for angular data.

Uses torch when available (preserves tensor type), falls back to numpy.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

# Optional torch support
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def _is_torch_tensor(x):
    """Check if x is a torch tensor."""
    return HAS_TORCH and isinstance(x, torch.Tensor)


def _normalize_axis(axis, dim):
    """Normalize axis/dim parameter."""
    return dim if dim is not None else axis


def _ensure_more_than_2d(data) -> None:
    """Ensure data has at least 2 dimensions."""
    ndim = data.ndim if hasattr(data, "ndim") else np.asarray(data).ndim
    assert ndim >= 2, (
        f"Input must be at least 2 dimensional with batch dimension as first axis, got {ndim}"
    )


def _check_angle_units(angles) -> None:
    """Check if angles might be in degrees and warn user."""
    if _is_torch_tensor(angles):
        max_val = torch.max(torch.abs(angles)).item()
    else:
        max_val = np.max(np.abs(angles))

    if max_val > 2 * np.pi:
        warnings.warn(
            f"Maximum angle value is {max_val:.2f} (>2π). "
            f"Consider using radians or angle wrapping.",
            UserWarning,
        )


def describe_circular(
    angles,
    values,
    axis: int = -1,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    funcs: Union[List[str], str] = [
        "circular_mean",
        "circular_concentration",
        "circular_skewness",
        "circular_kurtosis",
    ],
    device=None,
    batch_size: int = -1,
) -> Tuple[np.ndarray, List[str]]:
    """Compute various circular descriptive statistics.

    Parameters
    ----------
    angles : array-like
        Input angles in radians with batch dimension as first axis
    values : array-like
        Histogram values for each angle (must match angles shape)
    axis : int, default=-1
        Deprecated. Use dim instead
    dim : int or tuple of ints, optional
        Dimension(s) along which to compute statistics
    keepdims : bool, default=False
        Whether to keep reduced dimensions
    funcs : list of str or "all"
        Circular statistical functions to compute
    device : optional
        Device for torch tensors (ignored for numpy)
    batch_size : int, default=-1
        Batch size for processing (currently unused)

    Returns
    -------
    Tuple[ndarray or Tensor, List[str]]
        Computed circular statistics and their names
    """
    dim = _normalize_axis(axis, dim)
    dim = (dim,) if isinstance(dim, int) else tuple(dim) if dim is not None else None

    func_names = funcs
    func_candidates = {
        "circular_mean": circular_mean,
        "circular_concentration": circular_concentration,
        "circular_skewness": circular_skewness,
        "circular_kurtosis": circular_kurtosis,
    }

    if funcs == "all":
        _funcs = list(func_candidates.values())
        func_names = list(func_candidates.keys())
    else:
        _funcs = [func_candidates[ff] for ff in func_names]

    calculated = [ff(angles, values, dim=dim, keepdims=keepdims) for ff in _funcs]

    if _is_torch_tensor(angles):
        return torch.stack(calculated, dim=-1), func_names
    else:
        return np.stack(calculated, axis=-1), func_names


def circular_mean(
    angles,
    values,
    axis: int = -1,
    dim: int = None,
    batch_size: int = None,
    keepdims: bool = False,
):
    """Compute circular mean of angles weighted by histogram values.

    Parameters
    ----------
    angles : array-like
        Input angles in radians with batch dimension as first axis
    values : array-like
        Histogram values for each angle (must match angles shape)
    axis : int, default=-1
        Axis along which to compute mean (deprecated, use dim)
    dim : int, optional
        Dimension along which to compute mean
    batch_size : int, optional
        Batch size for processing (currently unused)
    keepdims : bool, default=False
        Whether to keep reduced dimensions

    Returns
    -------
    ndarray or Tensor
        Circular mean in range [0, 2π]
    """
    _ensure_more_than_2d(angles)
    _check_angle_units(angles)

    dim = _normalize_axis(axis, dim)

    if _is_torch_tensor(angles):
        assert angles.shape == values.shape, (
            f"angles shape {angles.shape} must match values shape {values.shape}"
        )
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        cos_component = torch.sum(values * cos_angles, dim=dim, keepdim=True)
        sin_component = torch.sum(values * sin_angles, dim=dim, keepdim=True)
        value_sum = torch.sum(values, dim=dim, keepdim=True)
        cos_component = cos_component / value_sum
        sin_component = sin_component / value_sum

        mean_angle = torch.atan2(sin_component, cos_component)
        mean_angle = torch.where(mean_angle < 0, mean_angle + 2 * np.pi, mean_angle)
        return mean_angle if keepdims else mean_angle.squeeze(dim)
    else:
        angles = np.asarray(angles)
        values = np.asarray(values)
        assert angles.shape == values.shape, (
            f"angles shape {angles.shape} must match values shape {values.shape}"
        )

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        cos_component = np.sum(values * cos_angles, axis=dim, keepdims=True)
        sin_component = np.sum(values * sin_angles, axis=dim, keepdims=True)
        value_sum = np.sum(values, axis=dim, keepdims=True)
        cos_component = cos_component / value_sum
        sin_component = sin_component / value_sum

        mean_angle = np.arctan2(sin_component, cos_component)
        mean_angle = np.where(mean_angle < 0, mean_angle + 2 * np.pi, mean_angle)
        return mean_angle if keepdims else np.squeeze(mean_angle, axis=dim)


def circular_concentration(
    angles,
    values,
    axis: int = -1,
    dim: int = None,
    batch_size: int = None,
    keepdims: bool = False,
):
    """Compute circular concentration (mean resultant length).

    Parameters
    ----------
    angles : array-like
        Input angles in radians with batch dimension as first axis
    values : array-like
        Histogram values for each angle (must match angles shape)
    axis : int, default=-1
        Axis along which to compute concentration (deprecated, use dim)
    dim : int, optional
        Dimension along which to compute concentration
    batch_size : int, optional
        Batch size for processing (currently unused)
    keepdims : bool, default=False
        Whether to keep reduced dimensions

    Returns
    -------
    ndarray or Tensor
        Concentration parameter in range [0, 1]
    """
    _ensure_more_than_2d(angles)
    _check_angle_units(angles)

    dim = _normalize_axis(axis, dim)

    if _is_torch_tensor(angles):
        assert angles.shape == values.shape
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        cos_component = torch.sum(values * cos_angles, dim=dim, keepdim=keepdims)
        sin_component = torch.sum(values * sin_angles, dim=dim, keepdim=keepdims)
        value_sum = torch.sum(values, dim=dim, keepdim=keepdims)
        return torch.sqrt(cos_component**2 + sin_component**2) / value_sum
    else:
        angles = np.asarray(angles)
        values = np.asarray(values)
        assert angles.shape == values.shape

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        cos_component = np.sum(values * cos_angles, axis=dim, keepdims=keepdims)
        sin_component = np.sum(values * sin_angles, axis=dim, keepdims=keepdims)
        value_sum = np.sum(values, axis=dim, keepdims=keepdims)
        return np.sqrt(cos_component**2 + sin_component**2) / value_sum


def circular_skewness(
    angles,
    values,
    axis: int = -1,
    dim: int = None,
    batch_size: int = None,
    keepdims: bool = False,
):
    """Compute circular skewness.

    Parameters
    ----------
    angles : array-like
        Input angles in radians with batch dimension as first axis
    values : array-like
        Histogram values for each angle (must match angles shape)
    axis : int, default=-1
        Axis along which to compute skewness (deprecated, use dim)
    dim : int, optional
        Dimension along which to compute skewness
    batch_size : int, optional
        Batch size for processing (currently unused)
    keepdims : bool, default=False
        Whether to keep reduced dimensions

    Returns
    -------
    ndarray or Tensor
        Circular skewness
    """
    _ensure_more_than_2d(angles)
    _check_angle_units(angles)

    dim = _normalize_axis(axis, dim)

    if _is_torch_tensor(angles):
        assert angles.shape == values.shape
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        cos_2angles = torch.cos(2 * angles)
        sin_2angles = torch.sin(2 * angles)

        value_sum = torch.sum(values, dim=dim, keepdim=True)
        c1 = torch.sum(values * cos_angles, dim=dim, keepdim=True) / value_sum
        s1 = torch.sum(values * sin_angles, dim=dim, keepdim=True) / value_sum
        c2 = torch.sum(values * cos_2angles, dim=dim, keepdim=True) / value_sum
        s2 = torch.sum(values * sin_2angles, dim=dim, keepdim=True) / value_sum

        skewness = (c2 * s1 - s2 * c1) / (1 - (c1**2 + s1**2)) ** (3 / 2)
        return skewness if keepdims else skewness.squeeze(dim)
    else:
        angles = np.asarray(angles)
        values = np.asarray(values)
        assert angles.shape == values.shape

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        cos_2angles = np.cos(2 * angles)
        sin_2angles = np.sin(2 * angles)

        value_sum = np.sum(values, axis=dim, keepdims=True)
        c1 = np.sum(values * cos_angles, axis=dim, keepdims=True) / value_sum
        s1 = np.sum(values * sin_angles, axis=dim, keepdims=True) / value_sum
        c2 = np.sum(values * cos_2angles, axis=dim, keepdims=True) / value_sum
        s2 = np.sum(values * sin_2angles, axis=dim, keepdims=True) / value_sum

        skewness = (c2 * s1 - s2 * c1) / (1 - (c1**2 + s1**2)) ** (3 / 2)
        return skewness if keepdims else np.squeeze(skewness, axis=dim)


def circular_kurtosis(
    angles,
    values,
    axis: int = -1,
    dim: int = None,
    batch_size: int = None,
    keepdims: bool = False,
):
    """Compute circular kurtosis.

    Parameters
    ----------
    angles : array-like
        Input angles in radians with batch dimension as first axis
    values : array-like
        Histogram values for each angle (must match angles shape)
    axis : int, default=-1
        Axis along which to compute kurtosis (deprecated, use dim)
    dim : int, optional
        Dimension along which to compute kurtosis
    batch_size : int, optional
        Batch size for processing (currently unused)
    keepdims : bool, default=False
        Whether to keep reduced dimensions

    Returns
    -------
    ndarray or Tensor
        Circular kurtosis
    """
    _ensure_more_than_2d(angles)
    _check_angle_units(angles)

    dim = _normalize_axis(axis, dim)

    if _is_torch_tensor(angles):
        assert angles.shape == values.shape
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        cos_2angles = torch.cos(2 * angles)
        sin_2angles = torch.sin(2 * angles)

        value_sum = torch.sum(values, dim=dim, keepdim=True)
        c1 = torch.sum(values * cos_angles, dim=dim, keepdim=True) / value_sum
        s1 = torch.sum(values * sin_angles, dim=dim, keepdim=True) / value_sum
        c2 = torch.sum(values * cos_2angles, dim=dim, keepdim=True) / value_sum
        s2 = torch.sum(values * sin_2angles, dim=dim, keepdim=True) / value_sum

        kurtosis = (c2 * c1 + s2 * s1) / (1 - (c1**2 + s1**2)) ** 2
        return kurtosis if keepdims else kurtosis.squeeze(dim)
    else:
        angles = np.asarray(angles)
        values = np.asarray(values)
        assert angles.shape == values.shape

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        cos_2angles = np.cos(2 * angles)
        sin_2angles = np.sin(2 * angles)

        value_sum = np.sum(values, axis=dim, keepdims=True)
        c1 = np.sum(values * cos_angles, axis=dim, keepdims=True) / value_sum
        s1 = np.sum(values * sin_angles, axis=dim, keepdims=True) / value_sum
        c2 = np.sum(values * cos_2angles, axis=dim, keepdims=True) / value_sum
        s2 = np.sum(values * sin_2angles, axis=dim, keepdims=True) / value_sum

        kurtosis = (c2 * c1 + s2 * s1) / (1 - (c1**2 + s1**2)) ** 2
        return kurtosis if keepdims else np.squeeze(kurtosis, axis=dim)


# EOF
