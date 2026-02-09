#!/usr/bin/env python3
# Timestamp: 2026-02-09
# File: /src/scitex/plt/utils/_colorbar.py

"""Enhanced colorbar utilities for scitex.plt — delegates styling to figrecipe."""

import matplotlib.pyplot as plt
from figrecipe._utils._colorbar import style_colorbar


def colorbar(mappable, ax=None, n_ticks=4, **kwargs):
    """Enhanced colorbar with better defaults for constrained_layout.

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

    defaults = {
        "fraction": 0.046,
        "pad": 0.04,
        "aspect": 20,
    }

    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value

    cbar = plt.colorbar(mappable, ax=ax, **kwargs)

    cbar.locator = MaxNLocator(nbins=n_ticks, min_n_ticks=2, prune="both")
    cbar.update_ticks()

    style_colorbar(cbar)

    if ax is not None:
        fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
        if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
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
        "shrink": 0.8,
        "aspect": 30,
    }

    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value

    cbar = fig.colorbar(mappable, ax=axes, location=location, **kwargs)

    style_colorbar(cbar)

    return cbar


# EOF
