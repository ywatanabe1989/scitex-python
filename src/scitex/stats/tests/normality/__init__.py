#!/usr/bin/env python3
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/normality/__init__.py

"""Normality tests for checking distributional assumptions."""

from ._test_ks import test_ks_1samp, test_ks_2samp
from ._test_shapiro import test_normality, test_shapiro

__all__ = [
    "test_shapiro",
    "test_normality",
    "test_ks_1samp",
    "test_ks_2samp",
]
