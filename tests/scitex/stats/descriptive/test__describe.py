#!/usr/bin/env python3
# Time-stamp: "2025-11-11"

"""Tests for scitex.stats.descriptive._describe module."""

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from scitex.stats.descriptive._describe import describe, verify_non_leakage


class TestDescribe:
    """Test describe function."""

    def test_basic_describe(self):
        """Test basic descriptive statistics."""
        x = torch.randn(10, 100)
        described, names = describe(x, dim=-1)

        assert described.shape == (10, 7), (
            f"Expected shape (10, 7), got {described.shape}"
        )
        assert len(names) == 7, "Should return 7 stat names"
        assert "nanmean" in names
        assert "nanstd" in names

    def test_with_nans(self):
        """Test with NaN values."""
        x = torch.randn(5, 50)
        x[0, 10:20] = float("nan")

        described, names = describe(x, dim=-1)

        # NaN-aware functions should handle NaN
        assert not torch.isnan(described).all(), "Should compute valid statistics"

    def test_different_dims(self):
        """Test with different dimensions."""
        x = torch.randn(4, 8, 16)

        # Reduce last dim
        desc1, _ = describe(x, dim=-1)
        assert desc1.shape == (4, 8, 7)

        # Reduce multiple dims
        desc2, _ = describe(x, dim=(1, 2))
        assert desc2.shape == (4, 7)

    def test_keepdims(self):
        """Test keepdims option."""
        x = torch.randn(5, 10, 20)

        described, _ = describe(x, dim=-1, keepdims=True)
        assert described.shape == (5, 10, 1, 7), "keepdims should preserve dimensions"

    def test_all_funcs(self):
        """Test with funcs='all'."""
        x = torch.randn(5, 20)

        described, names = describe(x, dim=-1, funcs="all")

        assert len(names) > 7, "Should return all available functions"
        assert "nanmax" in names
        assert "nanmin" in names
        assert "nancount" in names

    def test_custom_funcs(self):
        """Test with custom function list."""
        x = torch.randn(5, 20)

        custom_funcs = ["nanmean", "nanstd", "nanmax", "nanmin"]
        described, names = describe(x, dim=-1, funcs=custom_funcs)

        assert len(names) == 4
        assert names == custom_funcs

    def test_numpy_input(self):
        """Test with numpy array input."""
        x = np.random.randn(5, 20)

        described, names = describe(x, dim=-1)

        # New behavior: numpy in -> numpy out
        assert isinstance(described, np.ndarray)
        assert described.shape == (5, 7)


class TestVerifyNonLeakage:
    """Test verify_non_leakage function."""

    def test_no_leakage(self):
        """Test that no information leaks across batch."""
        x = torch.randn(10, 5, 20)

        # Should not raise error
        assert verify_non_leakage(x, dim=(1, 2))


class TestDescribeEdgeCases:
    """Test edge cases."""

    def test_all_nan(self):
        """Test with all NaN values."""
        x = torch.full((5, 10), float("nan"))

        described, names = describe(x, dim=-1)

        # Should return NaN but not crash
        assert described.shape == (5, 7)

    def test_single_value(self):
        """Test with single value."""
        x = torch.randn(5, 1)

        described, _ = describe(x, dim=-1)

        # Should handle single values
        assert described.shape == (5, 7)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_describe.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-20 15:05:08 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/desc/_describe.py
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
# - Computes comprehensive descriptive statistics on PyTorch tensors
# - Provides batch processing for large datasets
# - Validates non-leakage of information across samples
# - Combines multiple statistical measures into single function
# - Demonstrates statistical analysis with synthetic data
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
#   - Combined descriptive statistics results
# """
#
# """Imports"""
# import argparse
# from typing import List, Optional, Tuple, Union
#
# import numpy as np
# import scitex as stx
# import torch
# from scitex import logging
#
# from scitex.decorators import batch_fn, torch_fn
# from ._nan import (
#     nancount,
#     nankurtosis,
#     nanmax,
#     nanmean,
#     nanmin,
#     nanq25,
#     nanq50,
#     nanq75,
#     nanskewness,
#     nanstd,
#     nanvar,
# )
# from ._real import kurtosis, mean, q25, q50, q75, skewness, std
#
# logger = logging.getLogger(__name__)
#
# """Functions & Classes"""
#
#
# def verify_non_leakage(
#     x: torch.Tensor,
#     dim: Optional[Union[int, Tuple[int, ...]]] = None,
# ):
#     """Verifies that statistics computation doesn't leak information across samples."""
#     described, _ = describe(x, dim=(1, 2))
#     x_first = x[:1]
#     described_first, _ = describe(x_first, dim=dim)
#
#     assert described_first.shape == described[:1].shape, (
#         f"Shape mismatch: {described_first.shape} != {described[:1].shape}"
#     )
#
#     torch.testing.assert_close(
#         described_first,
#         described[:1],
#         rtol=1e-5,
#         atol=1e-8,
#         msg="Statistics leak information across samples",
#     )
#     return True
#
#
# @batch_fn
# @torch_fn
# def describe(
#     x: torch.Tensor,
#     axis: int = -1,
#     dim: Optional[Union[int, Tuple[int, ...]]] = None,
#     keepdims: bool = False,
#     funcs: Union[List[str], str] = [
#         "nanmean",
#         "nanstd",
#         "nankurtosis",
#         "nanskewness",
#         "nanq25",
#         "nanq50",
#         "nanq75",
#     ],
#     device: Optional[torch.device] = None,
#     batch_size: int = -1,
# ) -> Tuple[torch.Tensor, List[str]]:
#     """Compute descriptive statistics.
#
#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor with batch dimension as first axis
#     axis : int, default=-1
#         Deprecated. Use dim instead
#     dim : int or tuple of ints, optional
#         Dimension(s) along which to compute statistics
#     keepdims : bool, default=False
#         Whether to keep reduced dimensions
#     funcs : list of str or "all", default=["nanmean", "nanstd", "nankurtosis", "nanskewness", "nanq25", "nanq50", "nanq75"]
#         Statistical functions to compute
#     device : torch.device, optional
#         Device to use for computation
#     batch_size : int, default=-1
#         Batch size for processing (handled by decorator)
#
#     Returns
#     -------
#     Tuple[torch.Tensor, List[str]]
#         Computed statistics stacked along last dimension and their names
#     """
#
#     dim = axis if dim is None else dim
#     dim = (dim,) if isinstance(dim, int) else tuple(dim)
#
#     func_names = funcs
#     func_candidates = {
#         "mean": mean,
#         "std": std,
#         "kurtosis": kurtosis,
#         "skewness": skewness,
#         "q25": q25,
#         "q50": q50,
#         "q75": q75,
#         "nanmean": nanmean,
#         "nanstd": nanstd,
#         "nanvar": nanvar,
#         "nankurtosis": nankurtosis,
#         "nanskewness": nanskewness,
#         "nanq25": nanq25,
#         "nanq50": nanq50,
#         "nanq75": nanq75,
#         "nanmax": nanmax,
#         "nanmin": nanmin,
#         "nancount": nancount,
#     }
#
#     if funcs == "all":
#         _funcs = list(func_candidates.values())
#         func_names = list(func_candidates.keys())
#     else:
#         _funcs = [func_candidates[ff] for ff in func_names]
#
#     calculated = [ff(x, dim=dim, keepdims=keepdims) for ff in _funcs]
#     return torch.stack(calculated, dim=-1), func_names
#
#
# def main(args) -> int:
#     """Demonstrate comprehensive descriptive statistics with synthetic data."""
#     x = np.random.rand(10, 250, 16, 100, 100)
#
#     # Compute comprehensive statistics
#     described, method_names = describe(x[:10], dim=(-2, -1), batch_size=1)
#
#     # Store results
#     results = {
#         "input": x,
#         "described": described,
#         "method_names": method_names,
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
#         description="Demonstrate comprehensive descriptive statistics"
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
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/descriptive/_describe.py
# --------------------------------------------------------------------------------
