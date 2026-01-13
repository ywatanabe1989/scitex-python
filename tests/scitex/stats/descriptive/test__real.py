#!/usr/bin/env python3
# Time-stamp: "2025-11-11"

"""Tests for scitex.stats.descriptive._real module."""

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from scitex.stats.descriptive._real import (
    kurtosis,
    mean,
    q25,
    q50,
    q75,
    skewness,
    std,
    var,
    zscore,
)


class TestMean:
    """Test mean function."""

    def test_basic_mean(self):
        """Test basic mean calculation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean(x, dim=0)

        assert torch.isclose(result, torch.tensor(3.0))

    def test_2d_mean(self):
        """Test mean on 2D tensor."""
        x = torch.randn(5, 10)
        result = mean(x, dim=1)

        assert result.shape == (5,)


class TestStd:
    """Test std function."""

    def test_basic_std(self):
        """Test basic std calculation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = std(x, dim=0)

        # Should be around 1.58
        assert result > 0

    def test_constant_std(self):
        """Test std of constant values."""
        x = torch.tensor([2.0, 2.0, 2.0, 2.0])
        result = std(x, dim=0)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestZscore:
    """Test zscore function."""

    def test_basic_zscore(self):
        """Test basic z-score calculation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore(x, dim=0, keepdims=False)

        # Z-scores should have mean ~0 and std ~1
        assert torch.isclose(result.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(result.std(), torch.tensor(1.0), atol=1e-1)

    def test_zscore_keepdims(self):
        """Test z-score with keepdims."""
        x = torch.randn(5, 10)
        result = zscore(x, dim=1, keepdims=True)

        assert result.shape == (5, 10)


class TestSkewnessKurtosis:
    """Test skewness and kurtosis."""

    def test_symmetric_zero_skewness(self):
        """Test that symmetric distribution has ~0 skewness."""
        torch.manual_seed(42)
        x = torch.randn(1000)
        result = skewness(x, dim=0)

        # Should be close to 0 for normal distribution
        assert abs(result) < 0.5

    def test_kurtosis(self):
        """Test kurtosis calculation."""
        torch.manual_seed(42)
        x = torch.randn(1000)
        result = kurtosis(x, dim=0)

        # Excess kurtosis of normal is 0
        assert abs(result) < 1.0


class TestQuantiles:
    """Test quantile functions."""

    def test_q50_median(self):
        """Test that q50 equals median."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = q50(x, dim=0)

        assert torch.isclose(result, torch.tensor(3.0), atol=0.1)

    def test_q25(self):
        """Test 25th percentile."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = q25(x, dim=0)

        # Should be around 2.0
        assert 1.5 < result < 2.5

    def test_q75(self):
        """Test 75th percentile."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = q75(x, dim=0)

        # Should be around 4.0
        assert 3.5 < result < 4.5


class TestNumpyInput:
    """Test with numpy inputs."""

    def test_mean_numpy(self):
        """Test mean with numpy input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean(x, dim=0)

        # New behavior: numpy in -> numpy scalar/array out
        assert isinstance(result, (np.ndarray, np.floating))
        assert np.isclose(result, 3.0)

    def test_mean_numpy_2d(self):
        """Test mean with 2D numpy input."""
        x = np.random.randn(5, 10)
        result = mean(x, dim=1)

        # New behavior: numpy in -> numpy out, preserves shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_real.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-12-27 (refactored)"
# # File: scitex/stats/descriptive/_real.py
# """
# Descriptive statistics for real-valued data.
# 
# Uses torch when available (preserves tensor type), falls back to numpy.
# """
# 
# from __future__ import annotations
# 
# import os
# 
# import numpy as np
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# # Optional torch support
# try:
#     import torch
# 
#     HAS_TORCH = True
# except ImportError:
#     torch = None
#     HAS_TORCH = False
# 
# 
# def _is_torch_tensor(x):
#     """Check if x is a torch tensor."""
#     return HAS_TORCH and isinstance(x, torch.Tensor)
# 
# 
# def _normalize_axis(axis, dim):
#     """Normalize axis/dim parameter."""
#     return dim if dim is not None else axis
# 
# 
# # =============================================================================
# # Core Functions - Use torch when input is tensor, numpy otherwise
# # =============================================================================
# 
# 
# def mean(x, axis=-1, dim=None, keepdims=False):
#     """Compute mean along specified axis.
# 
#     Parameters
#     ----------
#     x : array-like
#         Input data (numpy array or torch tensor)
#     axis : int, default=-1
#         Axis along which to compute (deprecated, use dim)
#     dim : int or tuple, optional
#         Dimension(s) along which to compute
#     keepdims : bool, default=False
#         Keep reduced dimensions
# 
#     Returns
#     -------
#     ndarray or Tensor
#         Mean values (same type as input)
#     """
#     dim = _normalize_axis(axis, dim)
#     if _is_torch_tensor(x):
#         return x.mean(dim=dim, keepdim=keepdims)
#     return np.mean(np.asarray(x), axis=dim, keepdims=keepdims)
# 
# 
# def std(x, axis=-1, dim=None, keepdims=False):
#     """Compute standard deviation along specified axis."""
#     dim = _normalize_axis(axis, dim)
#     if _is_torch_tensor(x):
#         return x.std(dim=dim, keepdim=keepdims)
#     return np.std(np.asarray(x), axis=dim, keepdims=keepdims)
# 
# 
# def var(x, axis=-1, dim=None, keepdims=False):
#     """Compute variance along specified axis."""
#     dim = _normalize_axis(axis, dim)
#     if _is_torch_tensor(x):
#         return x.var(dim=dim, keepdim=keepdims)
#     return np.var(np.asarray(x), axis=dim, keepdims=keepdims)
# 
# 
# def zscore(x, axis=-1, dim=None, keepdims=True):
#     """Compute z-scores along specified axis."""
#     dim = _normalize_axis(axis, dim)
#     if _is_torch_tensor(x):
#         _mean = x.mean(dim=dim, keepdim=True)
#         _std = x.std(dim=dim, keepdim=True)
#         zscores = (x - _mean) / _std
#         return zscores if keepdims else zscores.squeeze(dim)
#     else:
#         x = np.asarray(x)
#         _mean = np.mean(x, axis=dim, keepdims=True)
#         _std = np.std(x, axis=dim, keepdims=True)
#         zscores = (x - _mean) / _std
#         if not keepdims and dim is not None:
#             zscores = np.squeeze(zscores, axis=dim)
#         return zscores
# 
# 
# def skewness(x, axis=-1, dim=None, keepdims=False):
#     """Compute skewness along specified axis."""
#     dim = _normalize_axis(axis, dim)
#     zscores = zscore(x, dim=dim, keepdims=True)
#     if _is_torch_tensor(x):
#         return torch.mean(torch.pow(zscores, 3.0), dim=dim, keepdim=keepdims)
#     return np.mean(np.power(zscores, 3.0), axis=dim, keepdims=keepdims)
# 
# 
# def kurtosis(x, axis=-1, dim=None, keepdims=False):
#     """Compute excess kurtosis along specified axis."""
#     dim = _normalize_axis(axis, dim)
#     zscores = zscore(x, dim=dim, keepdims=True)
#     if _is_torch_tensor(x):
#         return torch.mean(torch.pow(zscores, 4.0), dim=dim, keepdim=keepdims) - 3.0
#     return np.mean(np.power(zscores, 4.0), axis=dim, keepdims=keepdims) - 3.0
# 
# 
# def quantile(x, q, axis=-1, dim=None, keepdims=False):
#     """Compute quantile along specified axis.
# 
#     Parameters
#     ----------
#     x : array-like
#         Input data
#     q : float
#         Quantile to compute (0-100)
#     axis : int, default=-1
#         Axis along which to compute
#     dim : int or tuple, optional
#         Dimension(s) along which to compute
#     keepdims : bool, default=False
#         Keep reduced dimensions
# 
#     Returns
#     -------
#     ndarray or Tensor
#         Quantile values
#     """
#     dim = _normalize_axis(axis, dim)
# 
#     if _is_torch_tensor(x):
#         if isinstance(dim, (tuple, list)):
#             result = x
#             for d in sorted(dim, reverse=True):
#                 result = torch.quantile(result, q / 100, dim=d, keepdim=keepdims)
#             return result
#         return torch.quantile(x, q / 100, dim=dim, keepdim=keepdims)
#     else:
#         x = np.asarray(x)
#         if isinstance(dim, (tuple, list)):
#             result = x
#             for d in sorted(dim, reverse=True):
#                 result = np.quantile(result, q / 100, axis=d, keepdims=keepdims)
#             return result
#         return np.quantile(x, q / 100, axis=dim, keepdims=keepdims)
# 
# 
# def q25(x, axis=-1, dim=None, keepdims=False):
#     """Compute 25th percentile."""
#     return quantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# def q50(x, axis=-1, dim=None, keepdims=False):
#     """Compute 50th percentile (median)."""
#     return quantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# def q75(x, axis=-1, dim=None, keepdims=False):
#     """Compute 75th percentile."""
#     return quantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_real.py
# --------------------------------------------------------------------------------
