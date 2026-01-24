#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/timeseries/_TimeSeriesSlidingWindowSplit.py

"""Sliding window cross-validation for time series.

This module provides the TimeSeriesSlidingWindowSplit class which combines:
- Core splitting functionality from _sliding_window_core
- Visualization support from _sliding_window_plotting

For demo/example usage, see examples/ai/classification/sliding_window_demo.py
"""

from __future__ import annotations

from typing import Optional

from ._sliding_window_core import TimeSeriesSlidingWindowSplitCore
from ._sliding_window_plotting import SlidingWindowPlottingMixin

__all__ = ["TimeSeriesSlidingWindowSplit"]


class TimeSeriesSlidingWindowSplit(
    SlidingWindowPlottingMixin, TimeSeriesSlidingWindowSplitCore
):
    """Sliding window cross-validation for time series.

    Creates train/test windows that slide through time with configurable behavior.

    Parameters
    ----------
    window_size : int, optional
        Size of training window (ignored if expanding_window=True or n_splits is set).
        Required if n_splits is None.
    step_size : int, optional
        Step between windows (overridden if overlapping_tests=False)
    test_size : int, optional
        Size of test window. Required if n_splits is None.
    gap : int, default=0
        Number of samples to skip between train and test windows
    val_ratio : float, default=0.0
        Ratio of validation set from training window
    random_state : int, optional
        Random seed for reproducibility
    overlapping_tests : bool, default=False
        If False, automatically sets step_size=test_size to ensure each sample
        is tested exactly once (like K-fold for time series)
    expanding_window : bool, default=False
        If True, training window grows to include all past data (like sklearn's
        TimeSeriesSplit). If False, uses fixed sliding window of size window_size.
    undersample : bool, default=False
        If True, balance classes in training sets by randomly undersampling
        the majority class to match the minority class count. Temporal order
        is maintained. Requires y labels in split().
    n_splits : int, optional
        Number of splits to generate. If specified, window_size and test_size
        are automatically calculated to create exactly n_splits folds.
        Cannot be used together with manual window_size/test_size specification.

    Examples
    --------
    >>> from scitex.ai.classification import TimeSeriesSlidingWindowSplit
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> timestamps = np.arange(100)
    >>>
    >>> # Fixed window, non-overlapping tests (default)
    >>> swcv = TimeSeriesSlidingWindowSplit(window_size=50, test_size=10, gap=5)
    >>> for train_idx, test_idx in swcv.split(X, y, timestamps):
    ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    >>>
    >>> # Expanding window (use all past data)
    >>> swcv = TimeSeriesSlidingWindowSplit(
    ...     window_size=50, test_size=10, gap=5, expanding_window=True
    ... )
    >>> for train_idx, test_idx in swcv.split(X, y, timestamps):
    ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")  # Train grows!
    >>>
    >>> # Using n_splits (automatically calculates window and test sizes)
    >>> swcv = TimeSeriesSlidingWindowSplit(
    ...     n_splits=5, gap=0, expanding_window=True, undersample=True
    ... )
    >>> for train_idx, test_idx in swcv.split(X, y, timestamps):
    ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    >>>
    >>> # Visualize splits
    >>> fig = swcv.plot_splits(X, y, timestamps)
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        val_ratio: float = 0.0,
        random_state: Optional[int] = None,
        overlapping_tests: bool = False,
        expanding_window: bool = False,
        undersample: bool = False,
        n_splits: Optional[int] = None,
    ):
        super().__init__(
            window_size=window_size,
            step_size=step_size,
            test_size=test_size,
            gap=gap,
            val_ratio=val_ratio,
            random_state=random_state,
            overlapping_tests=overlapping_tests,
            expanding_window=expanding_window,
            undersample=undersample,
            n_splits=n_splits,
        )


# EOF
