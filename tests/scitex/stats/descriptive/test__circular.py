#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-11"

"""Tests for scitex.stats.descriptive._circular module."""

import numpy as np
import pytest
torch = pytest.importorskip("torch")
from scitex.stats.descriptive._circular import (
    circular_mean, circular_concentration,
    circular_skewness, circular_kurtosis,
    describe_circular
)


class TestCircularMean:
    """Test circular_mean function."""

    def test_basic_circular_mean(self):
        """Test basic circular mean."""
        # Angles around 0
        angles = torch.tensor([[0.1, 0.2, -0.1, -0.2]])
        values = torch.ones_like(angles)

        result = circular_mean(angles, values, dim=-1)

        # Mean should be close to 0
        # Handle tensor result - may have batch dimension or be in [0, 2π]
        if result.numel() == 1:
            val = result.item()
            # Circular mean wraps around, so close to 0 or 2π
            assert abs(val) < 0.1 or abs(val - 2*np.pi) < 0.1, f"Expected ~0, got {val}"
        else:
            # Check all values close to 0 or 2π
            assert ((torch.abs(result) < 0.1) | (torch.abs(result - 2*np.pi) < 0.1)).all()

    def test_opposite_angles(self):
        """Test mean of opposite angles."""
        angles = torch.tensor([[0.0, np.pi]])
        values = torch.ones_like(angles)
        
        result = circular_mean(angles, values, dim=-1)
        
        # Could be 0 or pi depending on cancellation
        assert torch.isfinite(result)

    def test_weighted_mean(self):
        """Test weighted circular mean."""
        angles = torch.tensor([[0.0, np.pi/2]])
        values = torch.tensor([[3.0, 1.0]])  # Heavily weighted towards 0
        
        result = circular_mean(angles, values, dim=-1)
        
        # Should be closer to 0 than pi/2
        assert result < np.pi/4

    def test_range_0_to_2pi(self):
        """Test that result is in [0, 2π]."""
        angles = torch.rand(10, 20) * 2 * np.pi
        values = torch.ones_like(angles)
        
        result = circular_mean(angles, values, dim=-1)
        
        assert torch.all(result >= 0)
        assert torch.all(result < 2 * np.pi)


class TestCircularConcentration:
    """Test circular_concentration function."""

    def test_high_concentration(self):
        """Test high concentration for clustered angles."""
        # All angles close together
        angles = torch.tensor([[0.1, 0.2, 0.15, 0.12]])
        values = torch.ones_like(angles)
        
        result = circular_concentration(angles, values, dim=-1)
        
        # Should be close to 1
        assert result > 0.9

    def test_low_concentration(self):
        """Test low concentration for dispersed angles."""
        # Uniformly distributed angles
        angles = torch.linspace(0, 2*np.pi, 100).unsqueeze(0)
        values = torch.ones_like(angles)
        
        result = circular_concentration(angles, values, dim=-1)
        
        # Should be close to 0
        assert result < 0.2

    def test_range_0_to_1(self):
        """Test that concentration is in [0, 1]."""
        angles = torch.rand(10, 20) * 2 * np.pi
        values = torch.ones_like(angles)
        
        result = circular_concentration(angles, values, dim=-1)
        
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)


class TestCircularSkewness:
    """Test circular_skewness function."""

    def test_circular_skewness(self):
        """Test circular skewness calculation."""
        angles = torch.tensor([[0.5, 1.2, 2.1, 3.8, 4.9, 5.7]])
        values = torch.tensor([[1.0, 2.0, 1.5, 1.0, 3.0, 1.2]])
        
        result = circular_skewness(angles, values, dim=-1)
        
        # Should be a finite value
        assert torch.isfinite(result)


class TestCircularKurtosis:
    """Test circular_kurtosis function."""

    def test_circular_kurtosis(self):
        """Test circular kurtosis calculation."""
        angles = torch.tensor([[0.5, 1.2, 2.1, 3.8, 4.9, 5.7]])
        values = torch.tensor([[1.0, 2.0, 1.5, 1.0, 3.0, 1.2]])
        
        result = circular_kurtosis(angles, values, dim=-1)
        
        # Should be a finite value
        assert torch.isfinite(result)


class TestDescribeCircular:
    """Test describe_circular function."""

    def test_describe_all(self):
        """Test comprehensive circular statistics."""
        angles = torch.rand(5, 20) * 2 * np.pi
        values = torch.ones_like(angles)
        
        result, names = describe_circular(angles, values, dim=-1, funcs='all')
        
        assert result.shape == (5, 4)  # 4 circular stats
        assert len(names) == 4
        assert 'circular_mean' in names
        assert 'circular_concentration' in names

    def test_custom_funcs(self):
        """Test with custom function list."""
        angles = torch.rand(5, 20) * 2 * np.pi
        values = torch.ones_like(angles)
        
        custom_funcs = ['circular_mean', 'circular_concentration']
        result, names = describe_circular(angles, values, dim=-1, funcs=custom_funcs)
        
        assert result.shape == (5, 2)
        assert names == custom_funcs


class TestCircularEdgeCases:
    """Test edge cases."""

    def test_single_angle(self):
        """Test with single angle."""
        angles = torch.tensor([[0.5]])
        values = torch.ones_like(angles)
        
        result = circular_mean(angles, values, dim=-1)
        
        # Mean of single angle is itself
        assert torch.isclose(result, torch.tensor(0.5), atol=1e-5)

    def test_zero_weights(self):
        """Test with some zero weights."""
        angles = torch.tensor([[0.0, np.pi, np.pi/2]])
        values = torch.tensor([[1.0, 0.0, 1.0]])  # Middle angle has 0 weight
        
        result = circular_mean(angles, values, dim=-1)
        
        # Should only consider first and third angles
        assert torch.isfinite(result)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_circular.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-12-27 (refactored)"
# # File: scitex/stats/descriptive/_circular.py
# """
# Circular statistics for angular data.
# 
# Uses torch when available (preserves tensor type), falls back to numpy.
# """
# 
# from __future__ import annotations
# 
# import os
# import warnings
# from typing import List, Optional, Tuple, Union
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
# def _ensure_more_than_2d(data) -> None:
#     """Ensure data has at least 2 dimensions."""
#     ndim = data.ndim if hasattr(data, "ndim") else np.asarray(data).ndim
#     assert ndim >= 2, (
#         f"Input must be at least 2 dimensional with batch dimension as first axis, got {ndim}"
#     )
# 
# 
# def _check_angle_units(angles) -> None:
#     """Check if angles might be in degrees and warn user."""
#     if _is_torch_tensor(angles):
#         max_val = torch.max(torch.abs(angles)).item()
#     else:
#         max_val = np.max(np.abs(angles))
# 
#     if max_val > 2 * np.pi:
#         warnings.warn(
#             f"Maximum angle value is {max_val:.2f} (>2π). "
#             f"Consider using radians or angle wrapping.",
#             UserWarning,
#         )
# 
# 
# def describe_circular(
#     angles,
#     values,
#     axis: int = -1,
#     dim: Optional[Union[int, Tuple[int, ...]]] = None,
#     keepdims: bool = False,
#     funcs: Union[List[str], str] = [
#         "circular_mean",
#         "circular_concentration",
#         "circular_skewness",
#         "circular_kurtosis",
#     ],
#     device=None,
#     batch_size: int = -1,
# ) -> Tuple[np.ndarray, List[str]]:
#     """Compute various circular descriptive statistics.
# 
#     Parameters
#     ----------
#     angles : array-like
#         Input angles in radians with batch dimension as first axis
#     values : array-like
#         Histogram values for each angle (must match angles shape)
#     axis : int, default=-1
#         Deprecated. Use dim instead
#     dim : int or tuple of ints, optional
#         Dimension(s) along which to compute statistics
#     keepdims : bool, default=False
#         Whether to keep reduced dimensions
#     funcs : list of str or "all"
#         Circular statistical functions to compute
#     device : optional
#         Device for torch tensors (ignored for numpy)
#     batch_size : int, default=-1
#         Batch size for processing (currently unused)
# 
#     Returns
#     -------
#     Tuple[ndarray or Tensor, List[str]]
#         Computed circular statistics and their names
#     """
#     dim = _normalize_axis(axis, dim)
#     dim = (dim,) if isinstance(dim, int) else tuple(dim) if dim is not None else None
# 
#     func_names = funcs
#     func_candidates = {
#         "circular_mean": circular_mean,
#         "circular_concentration": circular_concentration,
#         "circular_skewness": circular_skewness,
#         "circular_kurtosis": circular_kurtosis,
#     }
# 
#     if funcs == "all":
#         _funcs = list(func_candidates.values())
#         func_names = list(func_candidates.keys())
#     else:
#         _funcs = [func_candidates[ff] for ff in func_names]
# 
#     calculated = [ff(angles, values, dim=dim, keepdims=keepdims) for ff in _funcs]
# 
#     if _is_torch_tensor(angles):
#         return torch.stack(calculated, dim=-1), func_names
#     else:
#         return np.stack(calculated, axis=-1), func_names
# 
# 
# def circular_mean(
#     angles,
#     values,
#     axis: int = -1,
#     dim: int = None,
#     batch_size: int = None,
#     keepdims: bool = False,
# ):
#     """Compute circular mean of angles weighted by histogram values.
# 
#     Parameters
#     ----------
#     angles : array-like
#         Input angles in radians with batch dimension as first axis
#     values : array-like
#         Histogram values for each angle (must match angles shape)
#     axis : int, default=-1
#         Axis along which to compute mean (deprecated, use dim)
#     dim : int, optional
#         Dimension along which to compute mean
#     batch_size : int, optional
#         Batch size for processing (currently unused)
#     keepdims : bool, default=False
#         Whether to keep reduced dimensions
# 
#     Returns
#     -------
#     ndarray or Tensor
#         Circular mean in range [0, 2π]
#     """
#     _ensure_more_than_2d(angles)
#     _check_angle_units(angles)
# 
#     dim = _normalize_axis(axis, dim)
# 
#     if _is_torch_tensor(angles):
#         assert angles.shape == values.shape, (
#             f"angles shape {angles.shape} must match values shape {values.shape}"
#         )
#         cos_angles = torch.cos(angles)
#         sin_angles = torch.sin(angles)
# 
#         cos_component = torch.sum(values * cos_angles, dim=dim, keepdim=True)
#         sin_component = torch.sum(values * sin_angles, dim=dim, keepdim=True)
#         value_sum = torch.sum(values, dim=dim, keepdim=True)
#         cos_component = cos_component / value_sum
#         sin_component = sin_component / value_sum
# 
#         mean_angle = torch.atan2(sin_component, cos_component)
#         mean_angle = torch.where(mean_angle < 0, mean_angle + 2 * np.pi, mean_angle)
#         return mean_angle if keepdims else mean_angle.squeeze(dim)
#     else:
#         angles = np.asarray(angles)
#         values = np.asarray(values)
#         assert angles.shape == values.shape, (
#             f"angles shape {angles.shape} must match values shape {values.shape}"
#         )
# 
#         cos_angles = np.cos(angles)
#         sin_angles = np.sin(angles)
# 
#         cos_component = np.sum(values * cos_angles, axis=dim, keepdims=True)
#         sin_component = np.sum(values * sin_angles, axis=dim, keepdims=True)
#         value_sum = np.sum(values, axis=dim, keepdims=True)
#         cos_component = cos_component / value_sum
#         sin_component = sin_component / value_sum
# 
#         mean_angle = np.arctan2(sin_component, cos_component)
#         mean_angle = np.where(mean_angle < 0, mean_angle + 2 * np.pi, mean_angle)
#         return mean_angle if keepdims else np.squeeze(mean_angle, axis=dim)
# 
# 
# def circular_concentration(
#     angles,
#     values,
#     axis: int = -1,
#     dim: int = None,
#     batch_size: int = None,
#     keepdims: bool = False,
# ):
#     """Compute circular concentration (mean resultant length).
# 
#     Parameters
#     ----------
#     angles : array-like
#         Input angles in radians with batch dimension as first axis
#     values : array-like
#         Histogram values for each angle (must match angles shape)
#     axis : int, default=-1
#         Axis along which to compute concentration (deprecated, use dim)
#     dim : int, optional
#         Dimension along which to compute concentration
#     batch_size : int, optional
#         Batch size for processing (currently unused)
#     keepdims : bool, default=False
#         Whether to keep reduced dimensions
# 
#     Returns
#     -------
#     ndarray or Tensor
#         Concentration parameter in range [0, 1]
#     """
#     _ensure_more_than_2d(angles)
#     _check_angle_units(angles)
# 
#     dim = _normalize_axis(axis, dim)
# 
#     if _is_torch_tensor(angles):
#         assert angles.shape == values.shape
#         cos_angles = torch.cos(angles)
#         sin_angles = torch.sin(angles)
# 
#         cos_component = torch.sum(values * cos_angles, dim=dim, keepdim=keepdims)
#         sin_component = torch.sum(values * sin_angles, dim=dim, keepdim=keepdims)
#         value_sum = torch.sum(values, dim=dim, keepdim=keepdims)
#         return torch.sqrt(cos_component**2 + sin_component**2) / value_sum
#     else:
#         angles = np.asarray(angles)
#         values = np.asarray(values)
#         assert angles.shape == values.shape
# 
#         cos_angles = np.cos(angles)
#         sin_angles = np.sin(angles)
# 
#         cos_component = np.sum(values * cos_angles, axis=dim, keepdims=keepdims)
#         sin_component = np.sum(values * sin_angles, axis=dim, keepdims=keepdims)
#         value_sum = np.sum(values, axis=dim, keepdims=keepdims)
#         return np.sqrt(cos_component**2 + sin_component**2) / value_sum
# 
# 
# def circular_skewness(
#     angles,
#     values,
#     axis: int = -1,
#     dim: int = None,
#     batch_size: int = None,
#     keepdims: bool = False,
# ):
#     """Compute circular skewness.
# 
#     Parameters
#     ----------
#     angles : array-like
#         Input angles in radians with batch dimension as first axis
#     values : array-like
#         Histogram values for each angle (must match angles shape)
#     axis : int, default=-1
#         Axis along which to compute skewness (deprecated, use dim)
#     dim : int, optional
#         Dimension along which to compute skewness
#     batch_size : int, optional
#         Batch size for processing (currently unused)
#     keepdims : bool, default=False
#         Whether to keep reduced dimensions
# 
#     Returns
#     -------
#     ndarray or Tensor
#         Circular skewness
#     """
#     _ensure_more_than_2d(angles)
#     _check_angle_units(angles)
# 
#     dim = _normalize_axis(axis, dim)
# 
#     if _is_torch_tensor(angles):
#         assert angles.shape == values.shape
#         cos_angles = torch.cos(angles)
#         sin_angles = torch.sin(angles)
#         cos_2angles = torch.cos(2 * angles)
#         sin_2angles = torch.sin(2 * angles)
# 
#         value_sum = torch.sum(values, dim=dim, keepdim=True)
#         c1 = torch.sum(values * cos_angles, dim=dim, keepdim=True) / value_sum
#         s1 = torch.sum(values * sin_angles, dim=dim, keepdim=True) / value_sum
#         c2 = torch.sum(values * cos_2angles, dim=dim, keepdim=True) / value_sum
#         s2 = torch.sum(values * sin_2angles, dim=dim, keepdim=True) / value_sum
# 
#         skewness = (c2 * s1 - s2 * c1) / (1 - (c1**2 + s1**2)) ** (3 / 2)
#         return skewness if keepdims else skewness.squeeze(dim)
#     else:
#         angles = np.asarray(angles)
#         values = np.asarray(values)
#         assert angles.shape == values.shape
# 
#         cos_angles = np.cos(angles)
#         sin_angles = np.sin(angles)
#         cos_2angles = np.cos(2 * angles)
#         sin_2angles = np.sin(2 * angles)
# 
#         value_sum = np.sum(values, axis=dim, keepdims=True)
#         c1 = np.sum(values * cos_angles, axis=dim, keepdims=True) / value_sum
#         s1 = np.sum(values * sin_angles, axis=dim, keepdims=True) / value_sum
#         c2 = np.sum(values * cos_2angles, axis=dim, keepdims=True) / value_sum
#         s2 = np.sum(values * sin_2angles, axis=dim, keepdims=True) / value_sum
# 
#         skewness = (c2 * s1 - s2 * c1) / (1 - (c1**2 + s1**2)) ** (3 / 2)
#         return skewness if keepdims else np.squeeze(skewness, axis=dim)
# 
# 
# def circular_kurtosis(
#     angles,
#     values,
#     axis: int = -1,
#     dim: int = None,
#     batch_size: int = None,
#     keepdims: bool = False,
# ):
#     """Compute circular kurtosis.
# 
#     Parameters
#     ----------
#     angles : array-like
#         Input angles in radians with batch dimension as first axis
#     values : array-like
#         Histogram values for each angle (must match angles shape)
#     axis : int, default=-1
#         Axis along which to compute kurtosis (deprecated, use dim)
#     dim : int, optional
#         Dimension along which to compute kurtosis
#     batch_size : int, optional
#         Batch size for processing (currently unused)
#     keepdims : bool, default=False
#         Whether to keep reduced dimensions
# 
#     Returns
#     -------
#     ndarray or Tensor
#         Circular kurtosis
#     """
#     _ensure_more_than_2d(angles)
#     _check_angle_units(angles)
# 
#     dim = _normalize_axis(axis, dim)
# 
#     if _is_torch_tensor(angles):
#         assert angles.shape == values.shape
#         cos_angles = torch.cos(angles)
#         sin_angles = torch.sin(angles)
#         cos_2angles = torch.cos(2 * angles)
#         sin_2angles = torch.sin(2 * angles)
# 
#         value_sum = torch.sum(values, dim=dim, keepdim=True)
#         c1 = torch.sum(values * cos_angles, dim=dim, keepdim=True) / value_sum
#         s1 = torch.sum(values * sin_angles, dim=dim, keepdim=True) / value_sum
#         c2 = torch.sum(values * cos_2angles, dim=dim, keepdim=True) / value_sum
#         s2 = torch.sum(values * sin_2angles, dim=dim, keepdim=True) / value_sum
# 
#         kurtosis = (c2 * c1 + s2 * s1) / (1 - (c1**2 + s1**2)) ** 2
#         return kurtosis if keepdims else kurtosis.squeeze(dim)
#     else:
#         angles = np.asarray(angles)
#         values = np.asarray(values)
#         assert angles.shape == values.shape
# 
#         cos_angles = np.cos(angles)
#         sin_angles = np.sin(angles)
#         cos_2angles = np.cos(2 * angles)
#         sin_2angles = np.sin(2 * angles)
# 
#         value_sum = np.sum(values, axis=dim, keepdims=True)
#         c1 = np.sum(values * cos_angles, axis=dim, keepdims=True) / value_sum
#         s1 = np.sum(values * sin_angles, axis=dim, keepdims=True) / value_sum
#         c2 = np.sum(values * cos_2angles, axis=dim, keepdims=True) / value_sum
#         s2 = np.sum(values * sin_2angles, axis=dim, keepdims=True) / value_sum
# 
#         kurtosis = (c2 * c1 + s2 * s1) / (1 - (c1**2 + s1**2)) ** 2
#         return kurtosis if keepdims else np.squeeze(kurtosis, axis=dim)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_circular.py
# --------------------------------------------------------------------------------
