#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 10:00:00 (ywatanabe)"
# File: /src/scitex/plt/utils/_colorbar.py
# ----------------------------------------

"""Enhanced colorbar utilities for better placement with scitex.plt"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ._units import mm_to_pt


# ============================================================================
# Constants for colorbar styling
# ============================================================================
COLORBAR_LINE_WIDTH_MM = 0.2
COLORBAR_TICK_LENGTH_MM = 0.8
COLORBAR_TICK_FONTSIZE = 6  # pt


def style_colorbar(cbar):
    """Apply publication-quality styling to a colorbar.

    Applies:
    - 0.2mm outline thickness
    - 0.8mm tick length
    - 6pt tick labels

    Parameters
    ----------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar to style

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The styled colorbar
    """
    line_width = mm_to_pt(COLORBAR_LINE_WIDTH_MM)
    tick_length = mm_to_pt(COLORBAR_TICK_LENGTH_MM)

    # Style the colorbar outline
    cbar.outline.set_linewidth(line_width)

    # Style the ticks
    cbar.ax.tick_params(
        width=line_width, length=tick_length, labelsize=COLORBAR_TICK_FONTSIZE
    )

    # Style the colorbar axis spines
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(line_width)

    return cbar


def colorbar(mappable, ax=None, n_ticks=4, **kwargs):
    """Enhanced colorbar function that ensures proper spacing.

    This function wraps matplotlib.pyplot.colorbar with better defaults
    to prevent overlap with axes when using constrained_layout.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The mappable whose colorbar is to be made (e.g., from imshow, scatter)
    ax : matplotlib.axes.Axes or list of Axes, optional
        Parent axes from which space for a new colorbar axes will be stolen.
    n_ticks : int, optional
        Number of ticks on the colorbar. Default is 4 to match main axes style.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib.pyplot.colorbar

    Returns
    -------
    colorbar : matplotlib.colorbar.Colorbar
        The colorbar instance
    """
    from matplotlib.ticker import MaxNLocator

    # Set better defaults for colorbar placement
    defaults = {
        "fraction": 0.046,  # Fraction of axes to use for colorbar
        "pad": 0.04,  # Padding between axes and colorbar
        "aspect": 20,  # Aspect ratio of colorbar
    }

    # Update defaults with any user-provided kwargs
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value

    # Create the colorbar
    cbar = plt.colorbar(mappable, ax=ax, **kwargs)

    # Limit number of ticks to match main axes style (3-4 ticks)
    cbar.locator = MaxNLocator(nbins=n_ticks, min_n_ticks=2, prune="both")
    cbar.update_ticks()

    # Apply publication-quality styling
    style_colorbar(cbar)

    # If using constrained_layout, ensure the figure updates
    if ax is not None:
        fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
        if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
            # Force a layout update
            fig.canvas.draw_idle()

    return cbar


def add_shared_colorbar(fig, axes, mappable, location="right", **kwargs):
    """Add a single colorbar shared by multiple axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes
    axes : array-like of Axes
        The axes that will share the colorbar
    mappable : matplotlib.cm.ScalarMappable
        The mappable whose colorbar is to be made
    location : {'right', 'bottom', 'left', 'top'}, optional
        Where to place the colorbar (default: 'right')
    **kwargs : dict
        Additional keyword arguments passed to fig.colorbar

    Returns
    -------
    colorbar : matplotlib.colorbar.Colorbar
        The colorbar instance
    """
    defaults = {
        "shrink": 0.8,  # Shrink colorbar to match axes height
        "aspect": 30,  # Make it thinner for shared colorbars
    }

    # Update defaults with any user-provided kwargs
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value

    # Create the shared colorbar
    cbar = fig.colorbar(mappable, ax=axes, location=location, **kwargs)

    # Apply publication-quality styling
    style_colorbar(cbar)

    return cbar


# EOF
