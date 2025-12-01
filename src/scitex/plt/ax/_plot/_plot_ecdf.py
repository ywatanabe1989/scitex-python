#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 13:10:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_plot/_plot_ecdf.py

"""Empirical Cumulative Distribution Function (ECDF) plotting."""

import warnings
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from scitex.pd._force_df import force_df as scitex_pd_force_df
from ....plt.utils import assert_valid_axis


def plot_ecdf(
    axis: Union[Axes, "AxisWrapper"],
    data: np.ndarray,
    **kwargs: Any,
) -> Tuple[Union[Axes, "AxisWrapper"], pd.DataFrame]:
    """Plot Empirical Cumulative Distribution Function (ECDF).

    The ECDF shows the proportion of data points less than or equal to each
    value, representing the empirical estimate of the cumulative distribution
    function.

    Parameters
    ----------
    axis : matplotlib.axes.Axes or AxisWrapper
        Matplotlib axis or scitex axis wrapper to plot on.
    data : array-like
        Data to compute and plot ECDF for. NaN values are automatically ignored.
    **kwargs : dict
        Additional arguments passed to plot function.

    Returns
    -------
    axis : matplotlib.axes.Axes or AxisWrapper
        The axes with the ECDF plot.
    df : pd.DataFrame
        DataFrame containing ECDF data with columns:
        - x: sorted data values
        - y: cumulative percentages (0-100)
        - n: total number of data points
        - x_step, y_step: step plot coordinates

    Examples
    --------
    >>> import numpy as np
    >>> import scitex as stx
    >>> data = np.random.randn(100)
    >>> fig, ax = stx.plt.subplots()
    >>> ax, df = stx.plt.ax.plot_ecdf(ax, data)
    """
    assert_valid_axis(axis, "First argument must be a matplotlib axis or scitex axis wrapper")

    # Flatten and remove NaN values
    data = np.hstack(data)

    # Warnings
    if np.isnan(data).any():
        warnings.warn("NaN value are ignored for ECDF plot.")
    data = data[~np.isnan(data)]
    nn = len(data)

    # Sort the data and compute the ECDF values
    data_sorted = np.sort(data)
    ecdf_perc = 100 * np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    # Create the pseudo x-axis for step plotting
    x_step = np.repeat(data_sorted, 2)[1:]
    y_step = np.repeat(ecdf_perc, 2)[:-1]

    # Plot the ECDF using steps
    axis.plot(x_step, y_step, drawstyle="steps-post", **kwargs)

    # Scatter the original data points
    axis.plot(data_sorted, ecdf_perc, marker=".", linestyle="none")

    # Set ylim, xlim, and aspect ratio
    axis.set_ylim(0, 100)
    axis.set_xlim(0, 1.0)

    # Create a DataFrame to hold the ECDF data
    df = scitex_pd_force_df(
        {
            "x": data_sorted,
            "y": ecdf_perc,
            "n": nn,
            "x_step": x_step,
            "y_step": y_step,
        }
    )

    return axis, df


# EOF
