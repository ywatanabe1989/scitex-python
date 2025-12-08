#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 16:05:00 (ywatanabe)"
# File: ./src/scitex/plt/utils/_get_actual_font.py

"""
Detect the actual font being used by matplotlib after fallback resolution.
"""


def get_actual_font_name():
    """
    Get the name of the font actually being used by matplotlib.

    This resolves the font fallback chain to determine which font
    matplotlib is actually rendering with, not just what was requested.

    Returns
    -------
    str
        Name of the actual font being used (e.g., "DejaVu Sans", "Arial")

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    >>> actual_font = get_actual_font_name()
    >>> print(actual_font)  # "DejaVu Sans" if Arial not installed
    """
    import matplotlib.font_manager as fm

    # Get the actual font file path using matplotlib's font resolution
    # FontProperties() with no arguments uses the current rcParams settings
    actual_font_path = fm.findfont(fm.FontProperties())

    # Find the font name from the font manager's font list
    font_manager = fm.fontManager
    for font in font_manager.ttflist:
        if font.fname == actual_font_path:
            # Return the proper font name (e.g., "Arial" not "arial")
            return font.name

    # Fallback: extract from filename if not found in font list
    import os

    font_filename = os.path.basename(actual_font_path)
    # Remove .ttf extension and try to clean up the name
    font_name = os.path.splitext(font_filename)[0]
    # Capitalize common font names for better readability
    if font_name.lower() == "arial":
        return "Arial"
    elif font_name.lower().startswith("dejavusans"):
        return "DejaVu Sans"
    return font_name


# EOF
