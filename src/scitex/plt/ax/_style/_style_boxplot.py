#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 20:00:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_style_boxplot.py

"""
Style boxplot elements with millimeter-based control.

Default values are loaded from SCITEX_STYLE.yaml via presets.py.
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt

from scitex.plt.styles.presets import SCITEX_STYLE

# Get defaults from centralized config
_DEFAULT_LINEWIDTH_MM = SCITEX_STYLE.get("trace_thickness_mm", 0.2)
_DEFAULT_FLIER_SIZE_MM = SCITEX_STYLE.get("marker_size_mm", 0.8)


def style_boxplot(
    boxplot_dict,
    linewidth_mm: float = None,
    flier_size_mm: float = None,
    median_color: str = "black",
    edge_color: str = "black",
    colors: Optional[list] = None,
    add_legend: bool = False,
    labels: Optional[list] = None,
):
    """Apply publication-quality styling to matplotlib boxplot elements.

    This function modifies boxplots to:
    - Set consistent line widths for all elements
    - Set median line to black for visibility
    - Set edge colors to black
    - Apply consistent outlier marker styling
    - Use scitex color palette by default for box fills

    Parameters
    ----------
    boxplot_dict : dict
        Dictionary returned by ax.boxplot().
    linewidth_mm : float, default 0.2
        Line width in millimeters for all elements.
    flier_size_mm : float, default 0.8
        Outlier (flier) marker size in millimeters.
    median_color : str, default "black"
        Color for the median line inside boxes.
    edge_color : str, default "black"
        Color for box edges, whiskers, and caps.
    colors : list, optional
        List of colors for each box fill. If None, uses scitex color palette.
    add_legend : bool, default False
        Whether to add a legend.
    labels : list, optional
        Labels for legend entries (required if add_legend=True).

    Returns
    -------
    boxplot_dict : dict
        The styled boxplot dictionary.

    Examples
    --------
    >>> import scitex as stx
    >>> import numpy as np
    >>> fig, ax = stx.plt.subplots()
    >>> box_data = [np.random.normal(0, 1, 100) for _ in range(4)]
    >>> bp = ax.boxplot(box_data, patch_artist=True)
    >>> stx.plt.ax.style_boxplot(bp, median_color="black")
    """
    from scitex.plt.utils import mm_to_pt
    from scitex.plt.color._PARAMS import HEX

    # Use centralized defaults if not specified
    if linewidth_mm is None:
        linewidth_mm = _DEFAULT_LINEWIDTH_MM
    if flier_size_mm is None:
        flier_size_mm = _DEFAULT_FLIER_SIZE_MM

    # Convert mm to points
    lw_pt = mm_to_pt(linewidth_mm)
    flier_size_pt = mm_to_pt(flier_size_mm)

    # Use scitex color palette by default
    if colors is None:
        colors = [
            HEX["blue"],
            HEX["red"],
            HEX["green"],
            HEX["yellow"],
            HEX["purple"],
            HEX["orange"],
            HEX["lightblue"],
            HEX["pink"],
        ]

    # Style box elements with line width
    for element_name in ["boxes", "whiskers", "caps"]:
        if element_name in boxplot_dict:
            for element in boxplot_dict[element_name]:
                element.set_linewidth(lw_pt)
                element.set_color(edge_color)

    # Style medians with specified color
    if "medians" in boxplot_dict:
        for median in boxplot_dict["medians"]:
            median.set_linewidth(lw_pt)
            median.set_color(median_color)

    # Style fliers (outliers) with marker size
    if "fliers" in boxplot_dict:
        for flier in boxplot_dict["fliers"]:
            flier.set_markersize(flier_size_pt)
            flier.set_markeredgewidth(lw_pt)
            flier.set_markeredgecolor(edge_color)
            flier.set_markerfacecolor("none")  # Open circles

    # Apply fill colors to boxes
    for i, box in enumerate(boxplot_dict.get("boxes", [])):
        color = colors[i % len(colors)]
        if hasattr(box, "set_facecolor"):
            box.set_facecolor(color)
        box.set_edgecolor(edge_color)

    # Add legend if requested
    if add_legend and labels is not None:
        # Create proxy artists for legend
        import matplotlib.patches as mpatches

        if colors is not None:
            legend_elements = [
                mpatches.Patch(
                    facecolor="none", edgecolor=color, linewidth=lw_pt, label=label
                )
                for color, label in zip(colors, labels)
            ]
        else:
            legend_elements = [
                mpatches.Patch(
                    facecolor="none", edgecolor="C0", linewidth=lw_pt, label=label
                )
                for label in labels
            ]
        # Get the axes from one of the box elements
        if boxplot_dict.get("boxes"):
            ax = boxplot_dict["boxes"][0].axes
            ax.legend(handles=legend_elements)

    return boxplot_dict


# EOF
