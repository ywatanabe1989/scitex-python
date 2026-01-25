#!/usr/bin/env python3
"""Scitex descriptive statistics module."""

from ._ci import ci
from ._circular import (
    circular_concentration,
    circular_kurtosis,
    circular_mean,
    circular_skewness,
    describe_circular,
)
from ._describe import describe, verify_non_leakage
from ._nan import (
    nanargmax,
    nanargmin,
    nancount,
    nancumprod,
    nancumsum,
    nankurtosis,
    nanmax,
    nanmean,
    nanmin,
    nanprod,
    nanq25,
    nanq50,
    nanq75,
    nanquantile,
    nanskewness,
    nanstd,
    nansum,
    nanvar,
    nanzscore,
)
from ._real import (
    kurtosis,
    mean,
    q25,
    q50,
    q75,
    quantile,
    skewness,
    std,
    var,
    zscore,
)

__all__ = [
    # Confidence interval
    "ci",
    # For Ordinal Distribution
    "describe",
    "kurtosis",
    "mean",
    "nanargmax",
    "nanargmin",
    "nancount",
    "nancumprod",
    "nancumsum",
    "nankurtosis",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanprod",
    "nanq25",
    "nanq50",
    "nanq75",
    "nanquantile",
    "nanskewness",
    "nanstd",
    "nansum",
    "nanvar",
    "nanzscore",
    "q25",
    "q50",
    "q75",
    "quantile",
    "skewness",
    "std",
    "var",
    "verify_non_leakage",
    "zscore",
    # For Circular Distribution
    "circular_mean",
    "circular_concentration",
    "circular_skewness",
    "circular_kurtosis",
    "describe_circular",
]
