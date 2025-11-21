#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 15:52:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_plot/_add_fitted_line.py

"""
Add fitted regression line to scatter plots.
"""

import numpy as np
from typing import Optional, Tuple


def add_fitted_line(
    ax,
    x,
    y,
    color: str = 'black',
    linestyle: str = '--',
    linewidth_mm: float = 0.2,
    label: Optional[str] = None,
    degree: int = 1,
) -> Tuple:
    """
    Add a fitted polynomial line to a scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : array-like
        X data
    y : array-like
        Y data
    color : str, optional
        Line color (default: 'black')
    linestyle : str, optional
        Line style (default: '--' for dashed)
    linewidth_mm : float, optional
        Line thickness in millimeters (default: 0.2mm)
    label : str, optional
        Label for the fitted line (default: None)
    degree : int, optional
        Polynomial degree for fitting (default: 1 for linear)

    Returns
    -------
    line : Line2D
        The fitted line object
    coeffs : np.ndarray
        Polynomial coefficients from np.polyfit

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
    >>> scatter = ax.scatter(x, y)
    >>> stx.plt.ax.add_fitted_line(ax, x, y)

    >>> # With custom styling
    >>> line, coeffs = stx.plt.ax.add_fitted_line(
    ...     ax, x, y,
    ...     color='blue',
    ...     linestyle='-',
    ...     label='Linear fit'
    ... )
    """
    from scitex.plt.utils import mm_to_pt

    # Convert data to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    poly_fn = np.poly1d(coeffs)

    # Generate fitted line points
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly_fn(x_fit)

    # Convert linewidth to points
    lw_pt = mm_to_pt(linewidth_mm)

    # Plot fitted line
    line = ax.plot(
        x_fit,
        y_fit,
        color=color,
        linestyle=linestyle,
        linewidth=lw_pt,
        label=label,
    )[0]

    return line, coeffs


# EOF
