#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/ai/classification/timeseries/_sliding_window_core.py

"""Core TimeSeriesSlidingWindowSplit class without visualization."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples

from scitex import logging

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesSlidingWindowSplitCore"]


class TimeSeriesSlidingWindowSplitCore(BaseCrossValidator):
    """Sliding window cross-validation for time series (core functionality).

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
        # Handle n_splits mode vs manual mode
        if n_splits is not None:
            self.n_splits_mode = True
            self._n_splits = n_splits
            self.window_size = window_size if window_size is not None else 50
            self.test_size = test_size if test_size is not None else 10
        else:
            if window_size is None or test_size is None:
                raise ValueError(
                    "Either n_splits OR (window_size AND test_size) must be specified"
                )
            self.n_splits_mode = False
            self._n_splits = None
            self.window_size = window_size
            self.test_size = test_size

        self.gap = gap
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.overlapping_tests = overlapping_tests
        self.expanding_window = expanding_window
        self.undersample = undersample

        # Handle step_size logic
        if not overlapping_tests:
            if step_size is not None and step_size < test_size:
                logger.warning(
                    f"overlapping_tests=False but step_size={step_size} < test_size={test_size}. "
                    f"Setting step_size=test_size={test_size}."
                )
                self.step_size = test_size
            elif step_size is None:
                self.step_size = test_size
                logger.info(
                    f"step_size not specified with overlapping_tests=False. "
                    f"Using step_size=test_size={test_size}."
                )
            else:
                self.step_size = step_size
        else:
            if step_size is None:
                self.step_size = max(1, test_size // 2)
                logger.info(
                    f"step_size not specified with overlapping_tests=True. "
                    f"Using step_size={self.step_size} (50% overlap)."
                )
            else:
                self.step_size = step_size

    def _undersample_indices(
        self, train_indices: np.ndarray, y: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """Undersample majority class to balance training set.

        Maintains temporal order of samples.

        Parameters
        ----------
        train_indices : ndarray
            Original training indices
        y : ndarray
            Full label array
        timestamps : ndarray
            Full timestamp array

        Returns
        -------
        ndarray
            Undersampled training indices (sorted by timestamp)
        """
        train_labels = y[train_indices]
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)

        if len(unique_classes) < 2:
            return train_indices

        min_count = class_counts.min()

        undersampled_indices = []
        for cls in unique_classes:
            cls_mask = train_labels == cls
            cls_train_indices = train_indices[cls_mask]

            if len(cls_train_indices) > min_count:
                selected = self.rng.choice(
                    cls_train_indices, size=min_count, replace=False
                )
                undersampled_indices.extend(selected)
            else:
                undersampled_indices.extend(cls_train_indices)

        undersampled_indices = np.array(undersampled_indices)
        temporal_order = np.argsort(timestamps[undersampled_indices])
        undersampled_indices = undersampled_indices[temporal_order]

        return undersampled_indices

    def _calculate_auto_sizes(self, n_samples: int) -> None:
        """Auto-calculate window and test sizes for n_splits mode."""
        if self.expanding_window:
            min_window_size = max(1, n_samples // (self._n_splits + 1))
            available_for_test = (
                n_samples - min_window_size - (self._n_splits * self.gap)
            )
            calculated_test_size = max(1, available_for_test // self._n_splits)

            self.window_size = min_window_size
            self.test_size = calculated_test_size
            self.step_size = calculated_test_size

            logger.info(
                f"n_splits={self._n_splits} with expanding_window: "
                f"Calculated window_size={self.window_size}, test_size={self.test_size}"
            )
        else:
            available = n_samples - (self._n_splits * self.gap)
            calculated_test_size = max(1, available // (self._n_splits + 1))
            calculated_window_size = calculated_test_size

            self.window_size = calculated_window_size
            self.test_size = calculated_test_size
            self.step_size = calculated_test_size

            logger.info(
                f"n_splits={self._n_splits} with fixed window: "
                f"Calculated window_size={self.window_size}, test_size={self.test_size}"
            )

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate sliding window splits.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target variable
        timestamps : array-like, shape (n_samples,), optional
            Timestamps for temporal ordering. If None, uses sequential order
        groups : array-like, shape (n_samples,), optional
            Group labels (not used in this splitter)

        Yields
        ------
        train : ndarray
            Training set indices
        test : ndarray
            Test set indices
        """
        if timestamps is None:
            timestamps = np.arange(len(X))

        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        time_order = np.argsort(timestamps)
        sorted_indices = indices[time_order]

        if self.n_splits_mode:
            self._calculate_auto_sizes(n_samples)

        if self.expanding_window:
            yield from self._split_expanding(n_samples, sorted_indices, y, timestamps)
        else:
            yield from self._split_fixed(n_samples, sorted_indices, y, timestamps)

    def _split_expanding(
        self,
        n_samples: int,
        sorted_indices: np.ndarray,
        y: Optional[np.ndarray],
        timestamps: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        min_train_size = self.window_size
        total_min = min_train_size + self.gap + self.test_size

        if n_samples < total_min:
            logger.warning(
                f"Not enough samples ({n_samples}) for even one split. "
                f"Need at least {total_min} samples."
            )
            return

        test_start_pos = min_train_size + self.gap

        while test_start_pos + self.test_size <= n_samples:
            test_end_pos = test_start_pos + self.test_size
            train_end_pos = test_start_pos - self.gap
            train_indices = sorted_indices[0:train_end_pos]
            test_indices = sorted_indices[test_start_pos:test_end_pos]

            if self.undersample and y is not None:
                train_indices = self._undersample_indices(train_indices, y, timestamps)

            assert len(train_indices) > 0 and len(test_indices) > 0, "Empty window"
            yield train_indices, test_indices
            test_start_pos += self.step_size

    def _split_fixed(
        self,
        n_samples: int,
        sorted_indices: np.ndarray,
        y: Optional[np.ndarray],
        timestamps: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate fixed sliding window splits."""
        total_window = self.window_size + self.gap + self.test_size

        for start in range(0, n_samples - total_window + 1, self.step_size):
            train_end = start + self.window_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            train_indices = sorted_indices[start:train_end]
            test_indices = sorted_indices[test_start:test_end]

            if self.undersample and y is not None:
                train_indices = self._undersample_indices(train_indices, y, timestamps)

            assert len(train_indices) > 0 and len(test_indices) > 0, "Empty window"
            yield train_indices, test_indices

    def split_with_val(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate sliding window splits with validation set.

        The validation set comes after training but before test, maintaining
        temporal order: train < val < test.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target variable
        timestamps : array-like, shape (n_samples,), optional
            Timestamps for temporal ordering. If None, uses sequential order
        groups : array-like, shape (n_samples,), optional
            Group labels (not used in this splitter)

        Yields
        ------
        train : ndarray
            Training set indices
        val : ndarray
            Validation set indices
        test : ndarray
            Test set indices
        """
        if timestamps is None:
            timestamps = np.arange(len(X))

        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        time_order = np.argsort(timestamps)
        sorted_indices = indices[time_order]

        if self.n_splits_mode:
            self._calculate_auto_sizes(n_samples)

        val_size = int(self.window_size * self.val_ratio) if self.val_ratio > 0 else 0
        actual_train_size = self.window_size - val_size

        if self.expanding_window:
            yield from self._split_with_val_expanding(
                n_samples, sorted_indices, y, timestamps, val_size
            )
        else:
            yield from self._split_with_val_fixed(
                n_samples, sorted_indices, y, timestamps, val_size, actual_train_size
            )

    def _split_with_val_expanding(
        self,
        n_samples: int,
        sorted_indices: np.ndarray,
        y: Optional[np.ndarray],
        timestamps: np.ndarray,
        val_size: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate expanding window splits with validation."""
        min_train_size = self.window_size
        total_min = min_train_size + self.gap + self.test_size

        if n_samples < total_min:
            logger.warning(
                f"Not enough samples ({n_samples}) for even one split. "
                f"Need at least {total_min} samples."
            )
            return

        test_start_pos = min_train_size + self.gap

        while test_start_pos + self.test_size <= n_samples:
            test_end_pos = test_start_pos + self.test_size
            train_val_end_pos = test_start_pos - self.gap

            if val_size > 0:
                current_val_size = int(train_val_end_pos * self.val_ratio)
                train_end_pos = train_val_end_pos - current_val_size
                train_indices = sorted_indices[0:train_end_pos]
                val_indices = sorted_indices[train_end_pos:train_val_end_pos]
            else:
                train_indices = sorted_indices[0:train_val_end_pos]
                val_indices = np.array([])

            test_indices = sorted_indices[test_start_pos:test_end_pos]

            if self.undersample and y is not None:
                train_indices = self._undersample_indices(train_indices, y, timestamps)
                if len(val_indices) > 0:
                    val_indices = self._undersample_indices(val_indices, y, timestamps)

            assert len(train_indices) > 0 and len(test_indices) > 0, "Empty window"
            yield train_indices, val_indices, test_indices
            test_start_pos += self.step_size

    def _split_with_val_fixed(
        self,
        n_samples: int,
        sorted_indices: np.ndarray,
        y: Optional[np.ndarray],
        timestamps: np.ndarray,
        val_size: int,
        actual_train_size: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate fixed window splits with validation."""
        total_window = self.window_size + self.gap + self.test_size

        for start in range(0, n_samples - total_window + 1, self.step_size):
            train_end = start + actual_train_size
            val_start = train_end + (self.gap if val_size > 0 else 0)
            val_end = val_start + val_size
            test_start = val_end + self.gap if val_size > 0 else train_end + self.gap
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            train_indices = sorted_indices[start:train_end]
            val_indices = (
                sorted_indices[val_start:val_end] if val_size > 0 else np.array([])
            )
            test_indices = sorted_indices[test_start:test_end]

            if self.undersample and y is not None:
                train_indices = self._undersample_indices(train_indices, y, timestamps)
                if len(val_indices) > 0:
                    val_indices = self._undersample_indices(val_indices, y, timestamps)

            assert len(train_indices) > 0 and len(test_indices) > 0, "Empty window"
            yield train_indices, val_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Calculate number of splits.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            Training data (required to determine number of splits in manual mode)
        y : array-like, optional
            Not used
        groups : array-like, optional
            Not used

        Returns
        -------
        n_splits : int
            Number of splits. Returns -1 if X is None and not in n_splits mode.
        """
        if self.n_splits_mode:
            return self._n_splits

        if X is None:
            return -1

        n_samples = _num_samples(X)
        total_window = self.window_size + self.gap + self.test_size
        n_windows = (n_samples - total_window) // self.step_size + 1
        return max(0, n_windows)


# EOF
