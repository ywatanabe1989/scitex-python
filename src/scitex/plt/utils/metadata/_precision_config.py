#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_precision_config.py

"""
Numeric precision configuration.

This module defines precision settings for different measurement types
and provides a FixedFloat class for preserving decimal places in JSON output.
"""

from typing import Union

# Precision settings for JSON output
PRECISION = {
    "mm": 2,      # Millimeters: 0.01mm precision (10 microns)
    "inch": 3,    # Inches: 0.001 inch precision
    "position": 3, # Normalized position: 0.001 precision
    "lim": 2,     # Axis limits: 2 decimal places
    "linewidth": 2, # Line widths: 0.01 precision
}


class FixedFloat:
    """
    A float wrapper that preserves fixed decimal places in JSON output.

    Example: FixedFloat(0.25, 3) -> "0.250" in JSON
    """
    def __init__(self, value: float, precision: int):
        self.value = round(value, precision)
        self.precision = precision

    def __repr__(self):
        return f"{self.value:.{self.precision}f}"

    def __float__(self):
        return self.value


def _round_value(value: Union[float, int], precision: int, fixed: bool = False) -> Union[float, int, FixedFloat]:
    """
    Round a single value to specified precision.

    Parameters
    ----------
    value : float or int
        Value to round
    precision : int
        Number of decimal places
    fixed : bool
        If True, return FixedFloat with fixed decimal places (e.g., 0.250)
        If False, return float (e.g., 0.25)
    """
    if isinstance(value, int):
        if fixed:
            return FixedFloat(float(value), precision)
        return value
    if isinstance(value, float):
        if fixed:
            return FixedFloat(value, precision)
        return round(value, precision)
    return value


def _round_list(values: list, precision: int, fixed: bool = False) -> list:
    """Round all values in a list."""
    return [_round_value(v, precision, fixed) for v in values]
