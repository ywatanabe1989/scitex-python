#!/usr/bin/env python3
# Timestamp: 2026-01-26
# File: /home/ywatanabe/proj/scitex-python/src/scitex/stats/descriptive/_ci.py

"""Confidence interval computation."""

import numpy as np


def ci(xx, axis=None):
    """Compute 95% confidence interval.

    Calculates the confidence interval using the formula:
    CI = 1.96 * std / sqrt(n)

    Parameters
    ----------
    xx : array-like
        Input data array
    axis : int, optional
        Axis along which to compute CI (default: None, compute over entire array)

    Returns
    -------
    float or array
        95% confidence interval

    Example
    -------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> ci(data)
    1.2389...
    """
    indi = ~np.isnan(xx)
    return 1.96 * (xx[indi]).std(axis=axis) / np.sqrt(indi.sum())


# EOF
