#!/usr/bin/env python3
# Timestamp: "2025-12-27 (refactored)"
# File: scitex/stats/descriptive/_real.py
"""
Descriptive statistics for real-valued data.

Uses torch when available (preserves tensor type), falls back to numpy.
"""

from __future__ import annotations

import os

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


# =============================================================================
# Core Functions - Use torch when input is tensor, numpy otherwise
# =============================================================================


def mean(x, axis=-1, dim=None, keepdims=False):
    """Compute mean along specified axis.

    Parameters
    ----------
    x : array-like
        Input data (numpy array or torch tensor)
    axis : int, default=-1
        Axis along which to compute (deprecated, use dim)
    dim : int or tuple, optional
        Dimension(s) along which to compute
    keepdims : bool, default=False
        Keep reduced dimensions

    Returns
    -------
    ndarray or Tensor
        Mean values (same type as input)
    """
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        return x.mean(dim=dim, keepdim=keepdims)
    return np.mean(np.asarray(x), axis=dim, keepdims=keepdims)


def std(x, axis=-1, dim=None, keepdims=False):
    """Compute standard deviation along specified axis."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        return x.std(dim=dim, keepdim=keepdims)
    return np.std(np.asarray(x), axis=dim, keepdims=keepdims)


def var(x, axis=-1, dim=None, keepdims=False):
    """Compute variance along specified axis."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        return x.var(dim=dim, keepdim=keepdims)
    return np.var(np.asarray(x), axis=dim, keepdims=keepdims)


def zscore(x, axis=-1, dim=None, keepdims=True):
    """Compute z-scores along specified axis."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        _mean = x.mean(dim=dim, keepdim=True)
        _std = x.std(dim=dim, keepdim=True)
        zscores = (x - _mean) / _std
        return zscores if keepdims else zscores.squeeze(dim)
    else:
        x = np.asarray(x)
        _mean = np.mean(x, axis=dim, keepdims=True)
        _std = np.std(x, axis=dim, keepdims=True)
        zscores = (x - _mean) / _std
        if not keepdims and dim is not None:
            zscores = np.squeeze(zscores, axis=dim)
        return zscores


def skewness(x, axis=-1, dim=None, keepdims=False):
    """Compute skewness along specified axis."""
    dim = _normalize_axis(axis, dim)
    zscores = zscore(x, dim=dim, keepdims=True)
    if _is_torch_tensor(x):
        return torch.mean(torch.pow(zscores, 3.0), dim=dim, keepdim=keepdims)
    return np.mean(np.power(zscores, 3.0), axis=dim, keepdims=keepdims)


def kurtosis(x, axis=-1, dim=None, keepdims=False):
    """Compute excess kurtosis along specified axis."""
    dim = _normalize_axis(axis, dim)
    zscores = zscore(x, dim=dim, keepdims=True)
    if _is_torch_tensor(x):
        return torch.mean(torch.pow(zscores, 4.0), dim=dim, keepdim=keepdims) - 3.0
    return np.mean(np.power(zscores, 4.0), axis=dim, keepdims=keepdims) - 3.0


def quantile(x, q, axis=-1, dim=None, keepdims=False):
    """Compute quantile along specified axis.

    Parameters
    ----------
    x : array-like
        Input data
    q : float
        Quantile to compute (0-100)
    axis : int, default=-1
        Axis along which to compute
    dim : int or tuple, optional
        Dimension(s) along which to compute
    keepdims : bool, default=False
        Keep reduced dimensions

    Returns
    -------
    ndarray or Tensor
        Quantile values
    """
    dim = _normalize_axis(axis, dim)

    if _is_torch_tensor(x):
        if isinstance(dim, (tuple, list)):
            result = x
            for d in sorted(dim, reverse=True):
                result = torch.quantile(result, q / 100, dim=d, keepdim=keepdims)
            return result
        return torch.quantile(x, q / 100, dim=dim, keepdim=keepdims)
    else:
        x = np.asarray(x)
        if isinstance(dim, (tuple, list)):
            result = x
            for d in sorted(dim, reverse=True):
                result = np.quantile(result, q / 100, axis=d, keepdims=keepdims)
            return result
        return np.quantile(x, q / 100, axis=dim, keepdims=keepdims)


def q25(x, axis=-1, dim=None, keepdims=False):
    """Compute 25th percentile."""
    return quantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)


def q50(x, axis=-1, dim=None, keepdims=False):
    """Compute 50th percentile (median)."""
    return quantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)


def q75(x, axis=-1, dim=None, keepdims=False):
    """Compute 75th percentile."""
    return quantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)


# EOF
