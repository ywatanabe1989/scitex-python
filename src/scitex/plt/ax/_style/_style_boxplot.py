#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 14:30:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_style_boxplot.py

"""
Style boxplot elements with millimeter-based control.
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt


def style_boxplot(
    boxplot_dict,
    linewidth_mm: float = 0.8,
    colors: Optional[list] = None,
    add_legend: bool = False,
    labels: Optional[list] = None,
):
    """
    Apply consistent styling to matplotlib boxplot elements.

    Parameters
    ----------
    boxplot_dict : dict
        Dictionary returned by ax.boxplot()
    linewidth_mm : float, optional
        Line width in millimeters (default: 0.8mm for balanced appearance)
    colors : list, optional
        List of colors for each box. If None, uses default matplotlib colors.
    add_legend : bool, optional
        Whether to add a legend (default: False)
    labels : list, optional
        Labels for legend entries (required if add_legend=True)

    Returns
    -------
    boxplot_dict : dict
        The styled boxplot dictionary

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.NATURE_STYLE)
    >>> box_data = [np.random.normal(0, 1, 100) for _ in range(4)]
    >>> bp = ax.boxplot(box_data)
    >>> stx.ax.style_boxplot(bp, linewidth_mm=0.8, colors=['blue', 'red', 'green', 'orange'])
    """
    from scitex.plt.utils import mm_to_pt

    # Convert mm to points
    lw_pt = mm_to_pt(linewidth_mm)

    # Style box elements
    for element_name in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
        if element_name in boxplot_dict:
            for element in boxplot_dict[element_name]:
                element.set_linewidth(lw_pt)

    # Apply colors if provided
    if colors is not None:
        n_boxes = len(boxplot_dict.get('boxes', []))
        for i, box in enumerate(boxplot_dict.get('boxes', [])):
            color = colors[i % len(colors)]
            box.set_edgecolor(color)
            # Also color the associated whiskers, caps, and median
            if 'whiskers' in boxplot_dict:
                boxplot_dict['whiskers'][i*2].set_color(color)
                boxplot_dict['whiskers'][i*2+1].set_color(color)
            if 'caps' in boxplot_dict:
                boxplot_dict['caps'][i*2].set_color(color)
                boxplot_dict['caps'][i*2+1].set_color(color)
            if 'medians' in boxplot_dict and i < len(boxplot_dict['medians']):
                boxplot_dict['medians'][i].set_color(color)

    # Add legend if requested
    if add_legend and labels is not None:
        # Create proxy artists for legend
        import matplotlib.patches as mpatches
        if colors is not None:
            legend_elements = [
                mpatches.Patch(facecolor='none', edgecolor=color, linewidth=lw_pt, label=label)
                for color, label in zip(colors, labels)
            ]
        else:
            legend_elements = [
                mpatches.Patch(facecolor='none', edgecolor='C0', linewidth=lw_pt, label=label)
                for label in labels
            ]
        # Get the axes from one of the box elements
        if boxplot_dict.get('boxes'):
            ax = boxplot_dict['boxes'][0].axes
            ax.legend(handles=legend_elements)

    return boxplot_dict


# EOF
