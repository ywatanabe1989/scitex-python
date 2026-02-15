#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-09-21 20:45:00 (ywatanabe)"
# File: timeseries/__init__.py

"""
Time series cross-validation utilities for classification.

This module provides specialized cross-validation strategies for time series data,
ensuring proper temporal ordering and preventing data leakage.
"""

# Import timestamp normalizer
from ._normalize_timestamp import normalize_timestamp
from ._TimeSeriesBlockingSplit import TimeSeriesBlockingSplit
from ._TimeSeriesCalendarSplit import TimeSeriesCalendarSplit
from ._TimeSeriesMetadata import TimeSeriesMetadata
from ._TimeSeriesSlidingWindowSplit import TimeSeriesSlidingWindowSplit

# Import metadata and strategy
from ._TimeSeriesStrategy import TimeSeriesStrategy

# Import splitters
from ._TimeSeriesStratifiedSplit import TimeSeriesStratifiedSplit

__all__ = [
    # Main time series CV splitters
    "TimeSeriesStratifiedSplit",
    "TimeSeriesBlockingSplit",
    "TimeSeriesSlidingWindowSplit",
    "TimeSeriesCalendarSplit",
    # Support classes
    "TimeSeriesStrategy",
    "TimeSeriesMetadata",
    # Timestamp normalizer
    "normalize_timestamp",
]
