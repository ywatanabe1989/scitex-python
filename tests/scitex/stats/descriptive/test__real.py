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
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-20 14:56:35 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/desc/_real_dev.py
# # ----------------------------------------
# from __future__ import annotations
# import os
#
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# """
# Functionalities:
# - Computes descriptive statistics on PyTorch tensors
# - Provides mean, standard deviation, variance calculations
# - Calculates z-scores, skewness, kurtosis
# - Computes quantiles (25th, 50th, 75th percentiles)
# - Demonstrates statistical computations with synthetic data
#
# Dependencies:
# - packages:
#   - torch
#   - numpy
#   - scitex
#
# IO:
# - input-files:
#   - PyTorch tensor or numpy array
# - output-files:
#   - Descriptive statistics results
# """
#
# """Imports"""
# import argparse
#
# import numpy as np
# import scitex as stx
# import torch
# from scitex import logging
#
# from scitex.decorators import torch_fn
#
# logger = logging.getLogger(__name__)
#
# """Functions & Classes"""
#
#
# @torch_fn
# def mean(x, axis=-1, dim=None, keepdims=False):
#     return x.mean(dim, keepdims=keepdims)
#
#
# @torch_fn
# def std(x, axis=-1, dim=None, keepdims=False):
#     return x.std(dim, keepdims=keepdims)
#
#
# @torch_fn
# def var(x, axis=-1, dim=None, keepdims=False):
#     return x.var(dim, keepdims=keepdims)
#
#
# @torch_fn
# def zscore(x, axis=-1, dim=None, keepdims=True):
#     _mean = mean(x, dim=dim, keepdims=True)
#     _std = std(x, dim=dim, keepdims=True)
#     zscores = (x - _mean) / _std
#     return zscores if keepdims else zscores.squeeze(dim)
#
#
# @torch_fn
# def skewness(x, axis=-1, dim=None, keepdims=False):
#     zscores = zscore(x, axis=axis, keepdims=True)
#     return torch.mean(torch.pow(zscores, 3.0), dim=dim, keepdims=keepdims)
#
#
# @torch_fn
# def kurtosis(x, axis=-1, dim=None, keepdims=False):
#     zscores = zscore(x, axis=axis, keepdims=True)
#     return torch.mean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims) - 3.0
#
#
# @torch_fn
# def quantile(x, q, axis=-1, dim=None, keepdims=False):
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             x = torch.quantile(x, q / 100, dim=d, keepdims=keepdims)
#     else:
#         x = torch.quantile(x, q / 100, dim=dim, keepdims=keepdims)
#     return x
#
#
# @torch_fn
# def q25(x, axis=-1, dim=None, keepdims=False):
#     return quantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)
#
#
# @torch_fn
# def q50(x, axis=-1, dim=None, keepdims=False):
#     return quantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)
#
#
# @torch_fn
# def q75(x, axis=-1, dim=None, keepdims=False):
#     return quantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)
#
#
# def main(args) -> int:
#     """Demonstrate descriptive statistics functions with synthetic data."""
#     x = np.random.rand(4, 3, 2)
#
#     # Compute statistics
#     x_mean = mean(x)
#     x_std = std(x)
#     x_var = var(x)
#     x_skew = skewness(x)
#     x_kurt = kurtosis(x)
#     x_q25 = q25(x)
#     x_q50 = q50(x)
#     x_q75 = q75(x)
#
#     # Store results
#     results = {
#         "input": x,
#         "mean": x_mean,
#         "std": x_std,
#         "variance": x_var,
#         "skewness": x_skew,
#         "kurtosis": x_kurt,
#         "q25": x_q25,
#         "q50": x_q50,
#         "q75": x_q75,
#     }
#
#     for k, v in results.items():
#         if isinstance(v, (np.ndarray, torch.Tensor)):
#             print(f"\n{k}, Type: {type(v)}, Shape: {v.shape}, Values: {v}")
#         elif isinstance(v, list):
#             print(f"\n{k}, Type: {type(v)}, Length: {len(v)}, Values: {v}")
#         else:
#             print(f"\n{k}, Type: {type(v)}, Values: {v}")
#
#     return 0
#
#
# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate descriptive statistics functions"
#     )
#     args = parser.parse_args()
#     return args
#
#
# def run_main() -> None:
#     """Initialize scitex framework, run main function, and cleanup."""
#     global CONFIG, CC, sys, plt, rng
#     import sys
#
#     import matplotlib.pyplot as plt
#     import scitex as stx
#
#     args = parse_args()
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file=__FILE__,
#         sdir_suffix=None,
#         verbose=False,
#         agg=True,
#     )
#
#     exit_status = main(args)
#
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         message="",
#         exit_status=exit_status,
#     )
#
#
# if __name__ == "__main__":
#     run_main()
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_real.py
# --------------------------------------------------------------------------------
