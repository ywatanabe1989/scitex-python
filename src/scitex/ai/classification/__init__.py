#!/usr/bin/env python3
"""Classification utilities with unified API."""

# Import time series module
from . import timeseries

# Import other existing modules
from .Classifier import Classifier
from .CrossValidationExperiment import CrossValidationExperiment, quick_experiment

# Import reporters
from .reporters import ClassificationReporter, SingleTaskClassificationReporter

# Import time series CV utilities from submodule
from .timeseries import (
    TimeSeriesBlockingSplit,
    TimeSeriesCalendarSplit,
    TimeSeriesMetadata,
    TimeSeriesSlidingWindowSplit,
    TimeSeriesStrategy,
    TimeSeriesStratifiedSplit,
)

# Backward compatibility alias
CVExperiment = CrossValidationExperiment

__all__ = [
    # Reporters
    "ClassificationReporter",
    "SingleTaskClassificationReporter",
    # Classifier management
    "Classifier",
    # Cross-validation
    "CrossValidationExperiment",
    "CVExperiment",  # Alias
    "quick_experiment",
    # Time series module
    "timeseries",
    # Time series CV splitters (re-exported from timeseries module)
    "TimeSeriesStratifiedSplit",
    "TimeSeriesBlockingSplit",
    "TimeSeriesSlidingWindowSplit",
    "TimeSeriesCalendarSplit",
    "TimeSeriesStrategy",
    "TimeSeriesMetadata",
]
