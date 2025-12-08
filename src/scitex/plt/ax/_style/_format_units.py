#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 15:10:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_format_units.py

"""
Utility functions for formatting axis labels with proper unit notation.
"""

from typing import Optional


def format_label(label: str, unit: Optional[str] = None) -> str:
    """
    Format axis label with unit in brackets (publication standard).

    Parameters
    ----------
    label : str
        The label text (e.g., "Time", "Voltage")
    unit : str, optional
        The unit (e.g., "s", "mV", "Hz"). If None, returns label as-is.

    Returns
    -------
    str
        Formatted label with unit in brackets (e.g., "Time [s]")

    Examples
    --------
    >>> stx.ax.format_label("Time", "s")
    'Time [s]'

    >>> stx.ax.format_label("Voltage", "mV")
    'Voltage [mV]'

    >>> stx.ax.format_label("Count")
    'Count'

    >>> # Direct usage with axis
    >>> ax.set_xlabel(stx.ax.format_label("Time", "s"))
    >>> ax.set_ylabel(stx.ax.format_label("Amplitude", "mV"))

    Notes
    -----
    According to publication standards (Nature, Science, Cell), units should be
    enclosed in square brackets, not parentheses:
    - Correct: "Time [s]", "Voltage [mV]"
    - Incorrect: "Time (s)", "Voltage (mV)"
    """
    if unit is None or unit == "":
        return label
    return f"{label} [{unit}]"


def format_label_auto(text: str) -> str:
    """
    Automatically convert parentheses-style units to bracket-style.

    This function detects units in parentheses and converts them to brackets.

    Parameters
    ----------
    text : str
        Label text, possibly with units in parentheses

    Returns
    -------
    str
        Label text with units in brackets

    Examples
    --------
    >>> stx.ax.format_label_auto("Time (s)")
    'Time [s]'

    >>> stx.ax.format_label_auto("Voltage (mV)")
    'Voltage [mV]'

    >>> stx.ax.format_label_auto("Count")
    'Count'

    Notes
    -----
    This is useful for automatically correcting existing labels that use
    parentheses notation.
    """
    import re

    # Pattern to match units in parentheses at the end of the string
    # e.g., "Time (s)" or "Frequency (Hz)"
    pattern = r"\s*\(([^)]+)\)\s*$"

    match = re.search(pattern, text)
    if match:
        unit = match.group(1)
        label = text[: match.start()].strip()
        return f"{label} [{unit}]"

    return text


# EOF
