#!/usr/bin/env python3
# Timestamp: "2025-12-27 (refactored)"
# File: scitex/stats/descriptive/_nan.py
"""
NaN-aware descriptive statistics.

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
# NaN-aware Functions
# =============================================================================


def nanmax(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute maximum ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        min_value = torch.finfo(x.dtype).min
        if isinstance(dim, (tuple, list)):
            for d in sorted(dim, reverse=True):
                x = x.nan_to_num(min_value).max(dim=d, keepdim=keepdims)[0]
            return x
        return x.nan_to_num(min_value).max(dim=dim, keepdim=keepdims)[0]
    else:
        x = np.asarray(x)
        return np.nanmax(x, axis=dim, keepdims=keepdims)


def nanmin(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute minimum ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        max_value = torch.finfo(x.dtype).max
        if isinstance(dim, (tuple, list)):
            for d in sorted(dim, reverse=True):
                x = x.nan_to_num(max_value).min(dim=d, keepdim=keepdims)[0]
            return x
        return x.nan_to_num(max_value).min(dim=dim, keepdim=keepdims)[0]
    else:
        x = np.asarray(x)
        return np.nanmin(x, axis=dim, keepdims=keepdims)


def nansum(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute sum ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        return torch.nansum(x, dim=dim, keepdim=keepdims)
    return np.nansum(np.asarray(x), axis=dim, keepdims=keepdims)


def nanmean(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute mean ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        return torch.nanmean(x, dim=dim, keepdim=keepdims)
    return np.nanmean(np.asarray(x), axis=dim, keepdims=keepdims)


def nanvar(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute variance ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        tensor_mean = torch.nanmean(x, dim=dim, keepdim=True)
        return (x - tensor_mean).square().nanmean(dim=dim, keepdim=keepdims)
    else:
        x = np.asarray(x)
        return np.nanvar(x, axis=dim, keepdims=keepdims)


def nanstd(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute standard deviation ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        return torch.sqrt(nanvar(x, dim=dim, keepdims=keepdims))
    return np.nanstd(np.asarray(x), axis=dim, keepdims=keepdims)


def nanzscore(x, axis=-1, dim=None, batch_size=None, keepdims=True):
    """Compute z-scores ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        _mean = torch.nanmean(x, dim=dim, keepdim=True)
        _std = nanstd(x, dim=dim, keepdims=True)
        zscores = (x - _mean) / _std
        return zscores if keepdims else zscores.squeeze(dim)
    else:
        x = np.asarray(x)
        _mean = np.nanmean(x, axis=dim, keepdims=True)
        _std = np.nanstd(x, axis=dim, keepdims=True)
        zscores = (x - _mean) / _std
        if not keepdims and dim is not None:
            zscores = np.squeeze(zscores, axis=dim)
        return zscores


def nankurtosis(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute excess kurtosis ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    zscores = nanzscore(x, dim=dim, keepdims=True)
    if _is_torch_tensor(x):
        return torch.nanmean(torch.pow(zscores, 4.0), dim=dim, keepdim=keepdims) - 3.0
    return np.nanmean(np.power(zscores, 4.0), axis=dim, keepdims=keepdims) - 3.0


def nanskewness(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute skewness ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    zscores = nanzscore(x, dim=dim, keepdims=True)
    if _is_torch_tensor(x):
        return torch.nanmean(torch.pow(zscores, 3.0), dim=dim, keepdim=keepdims)
    return np.nanmean(np.power(zscores, 3.0), axis=dim, keepdims=keepdims)


def nanprod(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute product ignoring NaN values (treated as 1)."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        if isinstance(dim, (tuple, list)):
            for d in sorted(dim, reverse=True):
                x = x.nan_to_num(1).prod(dim=d, keepdim=keepdims)
            return x
        return x.nan_to_num(1).prod(dim=dim, keepdim=keepdims)
    else:
        x = np.asarray(x)
        return np.nanprod(x, axis=dim, keepdims=keepdims)


def nancumprod(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute cumulative product ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if isinstance(dim, (tuple, list)):
        raise ValueError("cumprod does not support multiple dimensions")
    if _is_torch_tensor(x):
        return x.nan_to_num(1).cumprod(dim=dim)
    return np.nancumprod(np.asarray(x), axis=dim)


def nancumsum(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute cumulative sum ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if isinstance(dim, (tuple, list)):
        raise ValueError("cumsum does not support multiple dimensions")
    if _is_torch_tensor(x):
        return x.nan_to_num(0).cumsum(dim=dim)
    return np.nancumsum(np.asarray(x), axis=dim)


def nanargmin(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute argmin ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        max_value = torch.finfo(x.dtype).max
        if isinstance(dim, (tuple, list)):
            for d in sorted(dim, reverse=True):
                x = x.nan_to_num(max_value).argmin(dim=d, keepdim=keepdims)
            return x
        return x.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdims)
    else:
        x = np.asarray(x)
        return (
            np.nanargmin(x, axis=dim, keepdims=keepdims)
            if keepdims
            else np.nanargmin(x, axis=dim)
        )


def nanargmax(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute argmax ignoring NaN values."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        min_value = torch.finfo(x.dtype).min
        if isinstance(dim, (tuple, list)):
            for d in sorted(dim, reverse=True):
                x = x.nan_to_num(min_value).argmax(dim=d, keepdim=keepdims)
            return x
        return x.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdims)
    else:
        x = np.asarray(x)
        return (
            np.nanargmax(x, axis=dim, keepdims=keepdims)
            if keepdims
            else np.nanargmax(x, axis=dim)
        )


def nanquantile(x, q, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute quantile ignoring NaN values.

    Parameters
    ----------
    x : array-like
        Input data
    q : float
        Quantile to compute (0-100)
    """
    dim = _normalize_axis(axis, dim)

    if _is_torch_tensor(x):
        if isinstance(dim, (tuple, list)):
            original_shape = x.shape
            dim_list = list(dim) if isinstance(dim, tuple) else dim
            dim_list = [d if d >= 0 else len(original_shape) + d for d in dim_list]
            keep_dims = [i for i in range(len(original_shape)) if i not in dim_list]
            perm_dims = keep_dims + dim_list
            x_perm = x.permute(perm_dims)
            new_shape = [original_shape[i] for i in keep_dims] + [-1]
            x_flat = x_perm.reshape(new_shape)
            mask = ~torch.isnan(x_flat)
            x_filtered = torch.where(mask, x_flat, torch.tensor(float("inf")))
            result = torch.quantile(x_filtered, q / 100, dim=-1, keepdim=keepdims)
            if keepdims:
                final_shape = list(original_shape)
                for d in dim_list:
                    final_shape[d] = 1
                result = result.reshape(final_shape)
            return result
        else:
            mask = ~torch.isnan(x)
            x_filtered = torch.where(mask, x, torch.tensor(float("inf")))
            return torch.quantile(x_filtered, q / 100, dim=dim, keepdim=keepdims)
    else:
        x = np.asarray(x)
        return np.nanquantile(x, q / 100, axis=dim, keepdims=keepdims)


def nanq25(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute 25th percentile ignoring NaN values."""
    return nanquantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)


def nanq50(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute 50th percentile (median) ignoring NaN values."""
    return nanquantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)


def nanq75(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Compute 75th percentile ignoring NaN values."""
    return nanquantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)


def nancount(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Count non-NaN values along specified dimensions."""
    dim = _normalize_axis(axis, dim)
    if _is_torch_tensor(x):
        mask = ~torch.isnan(x)
        if isinstance(dim, (tuple, list)):
            for d in sorted(dim, reverse=True):
                mask = mask.sum(dim=d, keepdim=keepdims)
            return mask
        return mask.sum(dim=dim, keepdim=keepdims)
    else:
        x = np.asarray(x)
        mask = ~np.isnan(x)
        return np.sum(mask, axis=dim, keepdims=keepdims)


# EOF
