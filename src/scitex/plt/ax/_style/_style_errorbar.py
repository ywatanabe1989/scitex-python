#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 14:50:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_style_errorbar.py

"""
Style error bar elements with millimeter-based control.
"""

from typing import Optional


def style_errorbar(
    errorbar_container,
    thickness_mm: float = 0.2,
    cap_width_mm: float = 0.8,
):
    """
    Apply consistent styling to matplotlib errorbar elements.

    Parameters
    ----------
    errorbar_container : ErrorbarContainer
        Container returned by ax.errorbar()
    thickness_mm : float, optional
        Line thickness for error bars in millimeters (default: 0.2mm)
    cap_width_mm : float, optional
        Cap width in millimeters (default: 0.8mm)

    Returns
    -------
    errorbar_container : ErrorbarContainer
        The styled errorbar container

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.NATURE_STYLE)
    >>> eb = ax.errorbar(x, y, yerr=yerr)
    >>> stx.plt.ax.style_errorbar(eb, thickness_mm=0.2, cap_width_mm=0.8)
    """
    from scitex.plt.utils import mm_to_pt

    # Convert mm to points
    lw_pt = mm_to_pt(thickness_mm)
    cap_width_pt = mm_to_pt(cap_width_mm)

    # Style the data line
    if errorbar_container[0] is not None:
        errorbar_container[0].set_linewidth(lw_pt)

    # Style the error bar lines
    if len(errorbar_container) > 2 and errorbar_container[2] is not None:
        for line_collection in errorbar_container[2]:
            if line_collection is not None:
                line_collection.set_linewidth(lw_pt)

    # Style the caps
    if len(errorbar_container) > 1 and errorbar_container[1] is not None:
        for cap in errorbar_container[1]:
            if cap is not None:
                cap.set_linewidth(lw_pt)  # Cap line thickness same as error bar
                # Set cap marker size (width)
                cap.set_markersize(cap_width_pt)

    return errorbar_container


# EOF
