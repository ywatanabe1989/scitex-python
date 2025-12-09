#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 20:50:45 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/plt/ax/_plot_statistical_shaded_line.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/plt/ax/_plot_statistical_shaded_line.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import numpy as np
import pandas as pd
from ....plt.utils import assert_valid_axis

from ._stx_shaded_line import stx_shaded_line as scitex_plt_plot_shaded_line


def _format_sample_size(values_2d):
    """Format sample size string, showing range if variable due to NaN.

    Parameters
    ----------
    values_2d : np.ndarray, shape (n_samples, n_points)
        2D array where sample count may vary per column due to NaN.

    Returns
    -------
    str
        Formatted sample size string, e.g., "20" or "18-20".
    """
    if values_2d.ndim == 1:
        return "1"

    # Count non-NaN values per column (timepoint)
    n_per_point = np.sum(~np.isnan(values_2d), axis=0)
    n_min, n_max = int(n_per_point.min()), int(n_per_point.max())

    if n_min == n_max:
        return str(n_min)
    else:
        return f"{n_min}-{n_max}"


def stx_line(axis, values_1d, xx=None, **kwargs):
    """
    Plot a simple line.

    Parameters
    ----------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis to plot on
    values_1d : array-like, shape (n_points,)
        1D array of y-values to plot
    xx : array-like, shape (n_points,), optional
        X coordinates for the data. If None, will use np.arange(len(values_1d))
    **kwargs
        Additional keyword arguments passed to axis.plot()

    Returns
    -------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis with the plot
    df : pandas.DataFrame
        DataFrame with x and y values
    """
    assert_valid_axis(
        axis, "First argument must be a matplotlib axis or scitex axis wrapper"
    )
    values_1d = np.asarray(values_1d)
    assert values_1d.ndim <= 2, f"Data must be 1D or 2D, got {values_1d.ndim}D"
    if xx is None:
        xx = np.arange(len(values_1d))
    else:
        xx = np.asarray(xx)
    assert len(xx) == len(values_1d), (
        f"xx length ({len(xx)}) must match values_1d length ({len(values_1d)})"
    )

    axis.plot(xx, values_1d, **kwargs)
    return axis, pd.DataFrame({"x": xx, "y": values_1d})


def stx_mean_std(axis, values_2d, xx=None, sd=1, **kwargs):
    """
    Plot mean line with standard deviation shading.

    Parameters
    ----------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis to plot on
    values_2d : array-like, shape (n_samples, n_points) or (n_points,)
        2D array where mean and std are calculated across axis=0 (samples).
        Can also be 1D for a single line without shading.
    xx : array-like, shape (n_points,), optional
        X coordinates for the data. If None, will use np.arange(n_points)
    sd : float, optional
        Number of standard deviations for the shaded region. Default is 1
    **kwargs
        Additional keyword arguments passed to stx_shaded_line()

    Returns
    -------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis with the plot
    """
    assert_valid_axis(
        axis, "First argument must be a matplotlib axis or scitex axis wrapper"
    )
    assert isinstance(sd, (int, float)), f"sd must be a number, got {type(sd)}"
    assert sd >= 0, f"sd must be non-negative, got {sd}"
    values_2d = np.asarray(values_2d)
    assert values_2d.ndim <= 2, f"Data must be 1D or 2D, got {values_2d.ndim}D"
    if xx is None:
        xx = np.arange(values_2d.shape[1] if values_2d.ndim > 1 else len(values_2d))
    else:
        xx = np.asarray(xx)
    expected_len = values_2d.shape[1] if values_2d.ndim > 1 else len(values_2d)
    assert len(xx) == expected_len, (
        f"xx length ({len(xx)}) must match values_2d length ({expected_len})"
    )

    if values_2d.ndim == 1:
        central = values_2d
        error = np.zeros_like(central)
    else:
        central = np.nanmean(values_2d, axis=0)
        error = np.nanstd(values_2d, axis=0) * sd

    y_lower = central - error
    y_upper = central + error

    if "label" in kwargs and kwargs["label"]:
        n_str = _format_sample_size(values_2d)
        kwargs["label"] = f"{kwargs['label']} ($n$={n_str})"

    return scitex_plt_plot_shaded_line(axis, xx, y_lower, central, y_upper, **kwargs)


def stx_mean_ci(axis, values_2d, xx=None, perc=95, **kwargs):
    """
    Plot mean line with confidence interval shading.

    Parameters
    ----------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis to plot on
    values_2d : array-like, shape (n_samples, n_points) or (n_points,)
        2D array where mean and percentiles are calculated across axis=0 (samples).
        Can also be 1D for a single line without shading.
    xx : array-like, shape (n_points,), optional
        X coordinates for the data. If None, will use np.arange(n_points)
    perc : float, optional
        Confidence interval percentage (0-100). Default is 95
    **kwargs
        Additional keyword arguments passed to stx_shaded_line()

    Returns
    -------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis with the plot
    """
    assert_valid_axis(
        axis, "First argument must be a matplotlib axis or scitex axis wrapper"
    )
    assert isinstance(perc, (int, float)), f"perc must be a number, got {type(perc)}"
    assert 0 <= perc <= 100, f"perc must be between 0 and 100, got {perc}"
    values_2d = np.asarray(values_2d)
    assert values_2d.ndim <= 2, f"Data must be 1D or 2D, got {values_2d.ndim}D"

    if xx is None:
        xx = np.arange(values_2d.shape[1] if values_2d.ndim > 1 else len(values_2d))
    else:
        xx = np.asarray(xx)

    expected_len = values_2d.shape[1] if values_2d.ndim > 1 else len(values_2d)
    assert len(xx) == expected_len, (
        f"xx length ({len(xx)}) must match values_2d length ({expected_len})"
    )

    if values_2d.ndim == 1:
        central = values_2d
        y_lower = central
        y_upper = central
    else:
        central = np.nanmean(values_2d, axis=0)
        # Calculate CI bounds
        alpha = 1 - perc / 100
        y_lower_perc = alpha / 2 * 100
        y_upper_perc = (1 - alpha / 2) * 100
        y_lower = np.nanpercentile(values_2d, y_lower_perc, axis=0)
        y_upper = np.nanpercentile(values_2d, y_upper_perc, axis=0)

    if "label" in kwargs and kwargs["label"]:
        n_str = _format_sample_size(values_2d)
        kwargs["label"] = f"{kwargs['label']} ($n$={n_str}, CI={perc}%)"

    return scitex_plt_plot_shaded_line(axis, xx, y_lower, central, y_upper, **kwargs)


def stx_median_iqr(axis, values_2d, xx=None, **kwargs):
    """
    Plot median line with interquartile range shading.

    Parameters
    ----------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis to plot on
    values_2d : array-like, shape (n_samples, n_points) or (n_points,)
        2D array where median and IQR are calculated across axis=0 (samples).
        Can also be 1D for a single line without shading.
    xx : array-like, shape (n_points,), optional
        X coordinates for the data. If None, will use np.arange(n_points)
    **kwargs
        Additional keyword arguments passed to stx_shaded_line()

    Returns
    -------
    axis : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axis with the plot
    """
    assert_valid_axis(
        axis, "First argument must be a matplotlib axis or scitex axis wrapper"
    )
    values_2d = np.asarray(values_2d)
    assert values_2d.ndim <= 2, f"Data must be 1D or 2D, got {values_2d.ndim}D"

    if xx is None:
        xx = np.arange(values_2d.shape[1] if values_2d.ndim > 1 else len(values_2d))
    else:
        xx = np.asarray(xx)

    expected_len = values_2d.shape[1] if values_2d.ndim > 1 else len(values_2d)
    assert len(xx) == expected_len, (
        f"xx length ({len(xx)}) must match values_2d length ({expected_len})"
    )

    if values_2d.ndim == 1:
        central = values_2d
        y_lower = central
        y_upper = central
    else:
        central = np.nanmedian(values_2d, axis=0)
        y_lower = np.nanpercentile(values_2d, 25, axis=0)
        y_upper = np.nanpercentile(values_2d, 75, axis=0)

    if "label" in kwargs and kwargs["label"]:
        n_str = _format_sample_size(values_2d)
        kwargs["label"] = f"{kwargs['label']} ($n$={n_str}, IQR)"

    return scitex_plt_plot_shaded_line(axis, xx, y_lower, central, y_upper, **kwargs)


# EOF
