#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_figure_mm.py

"""
Millimeter-based figure creation utilities for matplotlib.

This module provides functions to create matplotlib figures and axes with
precise millimeter-based control over dimensions, margins, and styling.
This is particularly useful for creating publication-quality figures that
need to meet specific size requirements (e.g., Nature, Science journals).
"""

__FILE__ = __file__

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._units import mm_to_inch, mm_to_pt


def create_figure_ax_mm(
    fig_width_mm: float = 35.0,
    fig_height_mm: float = 24.5,
    dpi: int = 300,
    *,
    left_margin_mm: float = 4.0,
    right_margin_mm: float = 2.0,
    bottom_margin_mm: float = 4.0,
    top_margin_mm: float = 2.0,
    style: Optional[Dict] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a Matplotlib figure and a single Axes with millimeter control.

    This function creates a figure with exact millimeter dimensions and positions
    the axes with specified margins. All measurements are in millimeters, allowing
    for precise control over the final output size.

    Parameters
    ----------
    fig_width_mm : float, optional
        Total figure width in millimeters (default: 35.0)
    fig_height_mm : float, optional
        Total figure height in millimeters (default: 24.5)
    dpi : int, optional
        Resolution in dots per inch for saving (default: 300)
    left_margin_mm : float, optional
        Left margin between figure edge and axis box, in millimeters (default: 4.0)
    right_margin_mm : float, optional
        Right margin between figure edge and axis box, in millimeters (default: 2.0)
    bottom_margin_mm : float, optional
        Bottom margin between figure edge and axis box, in millimeters (default: 4.0)
    top_margin_mm : float, optional
        Top margin between figure edge and axis box, in millimeters (default: 2.0)
    style : dict or None, optional
        Optional style specification dictionary. If provided, will be passed to
        apply_style_mm(). Expected keys:
        - 'axis_thickness_mm': Axis spine thickness in mm
        - 'trace_thickness_mm': Plot line thickness in mm
        - 'tick_length_mm': Tick mark length in mm
        - 'tick_thickness_mm': Tick mark thickness in mm
        - 'axis_font_size_pt': Axis label font size in points
        - 'tick_font_size_pt': Tick label font size in points

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure with specified dimensions
    ax : matplotlib.axes.Axes
        Created axes occupying the specified mm box with margins

    Examples
    --------
    Create a basic figure with default settings:

    >>> fig, ax = create_figure_ax_mm()
    >>> ax.plot([0, 1], [0, 1])
    >>> fig.savefig("test.png", dpi=300)

    Create a figure with custom dimensions and style:

    >>> style = {
    ...     'axis_thickness_mm': 0.2,
    ...     'trace_thickness_mm': 0.12,
    ...     'tick_length_mm': 0.8,
    ...     'tick_thickness_mm': 0.2,
    ...     'axis_font_size_pt': 8,
    ...     'tick_font_size_pt': 7,
    ... }
    >>> fig, ax = create_figure_ax_mm(
    ...     fig_width_mm=35,
    ...     fig_height_mm=24.5,
    ...     dpi=300,
    ...     left_margin_mm=4,
    ...     right_margin_mm=2,
    ...     bottom_margin_mm=4,
    ...     top_margin_mm=2,
    ...     style=style,
    ... )
    >>> ax.plot(x, y)
    >>> fig.savefig("test.tiff", dpi=300)

    Notes
    -----
    - The final saved figure will have exact physical dimensions when printed
      or embedded in documents (Word, PowerPoint, LaTeX)
    - The display size in browser/screen may appear different from the physical
      size, but the saved file will be correct
    - Use dpi=300 for high-quality publication figures
    """
    # Convert figure size from mm to inches (matplotlib uses inches)
    figsize_inch = (mm_to_inch(fig_width_mm), mm_to_inch(fig_height_mm))
    fig = plt.figure(figsize=figsize_inch, dpi=dpi)

    # Calculate axes position in figure coordinates [0–1]
    # The axes box occupies the space between margins
    axis_width_mm = fig_width_mm - left_margin_mm - right_margin_mm
    axis_height_mm = fig_height_mm - bottom_margin_mm - top_margin_mm

    # Convert to figure coordinates (0 to 1)
    left = left_margin_mm / fig_width_mm
    bottom = bottom_margin_mm / fig_height_mm
    width = axis_width_mm / fig_width_mm
    height = axis_height_mm / fig_height_mm

    # Create axes with exact position
    ax = fig.add_axes([left, bottom, width, height])

    # Apply styling if provided
    if style is not None:
        apply_style_mm(ax, style)

    return fig, ax


def apply_style_mm(ax: Axes, style: Dict) -> float:
    """
    Apply Nature-like style using millimeter-based settings.

    This function applies styling to matplotlib axes using millimeter and point
    measurements for precise control over visual elements. This is useful for
    meeting journal requirements (e.g., Nature, Science specifications).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to apply styling to
    style : dict
        Dictionary containing styling parameters. Supported keys:
        - 'axis_thickness_mm' (float): Spine line width in mm (default: 0.2)
        - 'trace_thickness_mm' (float): Plot line width in mm (default: 0.12)
        - 'tick_length_mm' (float): Tick mark length in mm (default: 0.8)
        - 'tick_thickness_mm' (float): Tick mark width in mm (default: 0.2)
        - 'axis_font_size_pt' (float): Axis label font size in points (default: 8)
        - 'tick_font_size_pt' (float): Tick label font size in points (default: 7)

    Returns
    -------
    float
        Trace line width in points, to be used with ax.plot(..., lw=trace_lw)

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> style = {
    ...     'axis_thickness_mm': 0.2,
    ...     'trace_thickness_mm': 0.12,
    ...     'tick_length_mm': 0.8,
    ...     'tick_thickness_mm': 0.2,
    ...     'axis_font_size_pt': 8,
    ...     'tick_font_size_pt': 7,
    ... }
    >>> trace_lw = apply_style_mm(ax, style)
    >>> ax.plot(x, y, lw=trace_lw)

    Notes
    -----
    - All thickness/width measurements are converted from mm to points
    - Font sizes are specified directly in points
    - The returned trace linewidth should be used when plotting to maintain
      consistent styling across all plot elements
    """
    # Convert spine thickness from mm to points
    axis_lw_pt = mm_to_pt(style.get("axis_thickness_mm", 0.2))
    for spine in ax.spines.values():
        spine.set_linewidth(axis_lw_pt)

    # Convert trace thickness from mm to points
    trace_lw_pt = mm_to_pt(style.get("trace_thickness_mm", 0.12))

    # Configure tick parameters (all mm values converted to points)
    ax.tick_params(
        direction="out",
        length=mm_to_pt(style.get("tick_length_mm", 0.8)),
        width=mm_to_pt(style.get("tick_thickness_mm", 0.2)),
    )

    # Apply font sizes (already in points, no conversion needed)
    axis_fs = style.get("axis_font_size_pt", 8)
    tick_fs = style.get("tick_font_size_pt", 7)

    ax.xaxis.label.set_fontsize(axis_fs)
    ax.yaxis.label.set_fontsize(axis_fs)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_fs)

    # Return trace linewidth for use in plotting
    return trace_lw_pt


if __name__ == "__main__":
    import numpy as np

    # Example usage
    print("Creating a test figure with mm control...")

    # Define Nature-like style
    nature_style = {
        "axis_thickness_mm": 0.2,
        "trace_thickness_mm": 0.12,
        "tick_length_mm": 0.8,
        "tick_thickness_mm": 0.2,
        "axis_font_size_pt": 8,
        "tick_font_size_pt": 7,
    }

    # Create figure with exact dimensions
    fig, ax = create_figure_ax_mm(
        fig_width_mm=35,
        fig_height_mm=35 * 0.7,  # Height is 70% of width
        dpi=300,
        left_margin_mm=4.0,
        right_margin_mm=2.0,
        bottom_margin_mm=4.0,
        top_margin_mm=2.0,
        style=nature_style,
    )

    # Plot some data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    trace_lw = mm_to_pt(nature_style["trace_thickness_mm"])
    ax.plot(x, y, color="tab:blue", lw=trace_lw)

    # Add labels
    ax.set_title("Test Plot with mm Control")
    ax.set_xlabel("X axis (rad)")
    ax.set_ylabel("Y axis")

    # Save the figure
    output_path = "/tmp/test_mm_control.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    print(f"Physical size: 35 mm × {35*0.7:.1f} mm")

# EOF
