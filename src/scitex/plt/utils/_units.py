#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_units.py

"""
Unit conversion utilities for matplotlib figure sizing.

This module provides conversion functions between millimeters, inches, and points,
which are commonly used for precise control of figure dimensions and styling in
publication-quality plots.
"""

__FILE__ = __file__

# Standard conversion constants
MM_PER_INCH = 25.4  # Millimeters per inch
PT_PER_INCH = 72.0  # Points per inch (PostScript standard)


def mm_to_inch(mm: float) -> float:
    """
    Convert millimeters to inches.

    Parameters
    ----------
    mm : float
        Length in millimeters

    Returns
    -------
    float
        Length in inches

    Examples
    --------
    >>> mm_to_inch(25.4)
    1.0
    >>> mm_to_inch(100)
    3.937...
    """
    return mm / MM_PER_INCH


def mm_to_pt(mm: float) -> float:
    """
    Convert millimeters to points (PostScript points).

    Parameters
    ----------
    mm : float
        Length in millimeters

    Returns
    -------
    float
        Length in points

    Examples
    --------
    >>> mm_to_pt(25.4)
    72.0
    >>> mm_to_pt(1.0)
    2.834...
    """
    return mm * PT_PER_INCH / MM_PER_INCH


def inch_to_mm(inch: float) -> float:
    """
    Convert inches to millimeters.

    Parameters
    ----------
    inch : float
        Length in inches

    Returns
    -------
    float
        Length in millimeters

    Examples
    --------
    >>> inch_to_mm(1.0)
    25.4
    """
    return inch * MM_PER_INCH


def pt_to_mm(pt: float) -> float:
    """
    Convert points to millimeters.

    Parameters
    ----------
    pt : float
        Length in points

    Returns
    -------
    float
        Length in millimeters

    Examples
    --------
    >>> pt_to_mm(72.0)
    25.4
    """
    return pt * MM_PER_INCH / PT_PER_INCH


if __name__ == "__main__":
    # Simple tests
    print("Unit Conversion Tests:")
    print("-" * 40)
    print(f"25.4 mm = {mm_to_inch(25.4):.2f} inches")
    print(f"1 inch = {inch_to_mm(1.0):.2f} mm")
    print(f"25.4 mm = {mm_to_pt(25.4):.2f} points")
    print(f"72 points = {pt_to_mm(72.0):.2f} mm")
    print("-" * 40)

# EOF
