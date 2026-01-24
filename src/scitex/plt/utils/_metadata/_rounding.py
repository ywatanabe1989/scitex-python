#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_rounding.py

"""
Rounding utilities for figure metadata.

Provides precision-controlled rounding for various measurement types
(mm, inch, position, etc.) to ensure consistent JSON output.
"""

from typing import List, Union

# Precision settings for JSON output
PRECISION = {
    "mm": 2,  # Millimeters: 0.01mm precision (10 microns)
    "inch": 3,  # Inches: 0.001 inch precision
    "position": 3,  # Normalized position: 0.001 precision
    "lim": 2,  # Axis limits: 2 decimal places
    "linewidth": 2,  # Line widths: 0.01 precision
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


def _round_value(
    value: Union[float, int], precision: int, fixed: bool = False
) -> Union[float, int, "FixedFloat"]:
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


def _round_list(values: List, precision: int, fixed: bool = False) -> List:
    """Round all values in a list."""
    return [_round_value(v, precision, fixed) for v in values]


def _round_dict(d: dict, precision_map: dict = None) -> dict:
    """
    Round all float values in a dict based on key-specific precision.

    Parameters
    ----------
    d : dict
        Dictionary to process
    precision_map : dict, optional
        Mapping of key patterns to precision values.
        Default uses PRECISION settings based on key names.
    """
    if precision_map is None:
        precision_map = {}

    result = {}
    for key, value in d.items():
        # Determine precision based on key name
        if "mm" in key.lower():
            prec = PRECISION["mm"]
        elif "inch" in key.lower():
            prec = PRECISION["inch"]
        elif "position" in key.lower() or key in ("left", "bottom", "right", "top"):
            prec = PRECISION["position"]
        elif "lim" in key.lower():
            prec = PRECISION["lim"]
        elif "width" in key.lower() and "line" in key.lower():
            prec = PRECISION["linewidth"]
        else:
            prec = precision_map.get(key, 3)  # Default 3 decimals

        if isinstance(value, dict):
            result[key] = _round_dict(value, precision_map)
        elif isinstance(value, list):
            result[key] = _round_list(value, prec)
        elif isinstance(value, float):
            result[key] = _round_value(value, prec)
        else:
            result[key] = value

    return result


# EOF
