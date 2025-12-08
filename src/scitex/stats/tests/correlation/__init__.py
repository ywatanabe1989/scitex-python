#!/usr/bin/env python3
"""
Correlation tests module.

Available tests:
- test_pearson: Pearson correlation (linear relationship)
"""

from ._test_pearson import test_pearson

__all__ = [
    "test_pearson",
]
