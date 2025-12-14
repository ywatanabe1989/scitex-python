#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _statistical.py - Statistical plot methods

"""Statistical plotting methods including line plots, box plots, and violin plots."""

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scitex.types import ArrayLike

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


class StatisticalPlotMixin:
    """Mixin for statistical plotting methods.

    Provides methods for:
    - Distribution plots (boxplot, violin)
    - Line plots with uncertainty (mean±std, mean±CI, median±IQR)
    - Histograms with bin alignment
    - Geometric shapes (rectangles, filled regions)
    """

    def stx_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> "Axes":
        """Draw a rectangle on the axes.

        Parameters
        ----------
        x : float
            X coordinate of the lower-left corner.
        y : float
            Y coordinate of the lower-left corner.
        width : float
            Width of the rectangle.
        height : float
            Height of the rectangle.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the rectangle function.

        Returns
        -------
        Axes
            The axes with the rectangle added.

        Examples
        --------
        >>> ax.stx_rectangle(0, 0, 1, 2, color='blue', alpha=0.5)
        """
        method_name = "stx_rectangle"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_rectangle(
                self._axis_mpl, x, y, width, height, **kwargs
            )

        tracked_dict = {"x": x, "y": y, "width": width, "height": height}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_fillv(
        self,
        starts: ArrayLike,
        ends: ArrayLike,
        *,
        color: str = "red",
        alpha: float = 0.2,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> "Axes":
        """Fill vertical spans between start and end positions.

        Parameters
        ----------
        starts : array-like
            Start x-coordinates of each span.
        ends : array-like
            End x-coordinates of each span.
        color : str, default 'red'
            Fill color.
        alpha : float, default 0.2
            Transparency level.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the fill function.

        Returns
        -------
        Axes
            The axes with the filled spans added.

        Examples
        --------
        >>> ax.stx_fillv([0, 2, 4], [1, 3, 5], color='green')
        """
        method_name = "stx_fillv"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_fillv(
                self._axis_mpl, starts, ends, color=color, alpha=alpha
            )

        tracked_dict = {"starts": starts, "ends": ends}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_box(
        self,
        data: Union[ArrayLike, Sequence[ArrayLike]],
        *,
        colors: Optional[List[str]] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Create a boxplot with SciTeX styling and tracking.

        Parameters
        ----------
        data : array-like or sequence of array-like
            Data for the boxplot. Can be a single array or list of arrays
            where each array represents a group.
        colors : list of str, optional
            Colors for each box. If None, uses default palette.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to `matplotlib.axes.Axes.boxplot`.

        Returns
        -------
        dict
            Dictionary mapping component names ('boxes', 'whiskers', etc.)
            to lists of Line2D or Patch artists.

        See Also
        --------
        stx_boxplot : Alias for this method.
        sns_boxplot : DataFrame-based boxplot.
        stx_violin : Violin plot alternative.

        Examples
        --------
        >>> ax.stx_box([data1, data2, data3], labels=['A', 'B', 'C'])
        >>> ax.stx_box(data, notch=True, patch_artist=True)
        """
        method_name = "stx_box"

        _data = data.copy()

        if kwargs.get("label"):
            n_per_group = [len(g) for g in data]
            n_min, n_max = min(n_per_group), max(n_per_group)
            n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
            kwargs["label"] = kwargs["label"] + f" ($n$={n_str})"

        if "patch_artist" not in kwargs:
            kwargs["patch_artist"] = True

        with self._no_tracking():
            result = self._axis_mpl.boxplot(data, **kwargs)

        n_per_group = [len(g) for g in data]
        tracked_dict = {"data": _data, "n": n_per_group}
        self._track(track, id, method_name, tracked_dict, None)

        from scitex.plt.ax import style_boxplot

        style_boxplot(result, colors=colors)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def hist(
        self,
        x: ArrayLike,
        *,
        bins: Union[int, str, ArrayLike] = 10,
        range: Optional[Tuple[float, float]] = None,
        align_bins: bool = True,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, "BarContainer"]:
        """Plot a histogram with optional bin alignment across multiple histograms.

        Parameters
        ----------
        x : array-like
            Input data for the histogram.
        bins : int, str, or array-like, default 10
            Number of bins, binning strategy ('auto', 'fd', etc.), or bin edges.
        range : tuple of float, optional
            Lower and upper range of the bins. If None, uses data range.
        align_bins : bool, default True
            When True, aligns bins across multiple histograms on the same axes.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to `matplotlib.axes.Axes.hist`.

        Returns
        -------
        tuple
            (counts, bin_edges, patches) from matplotlib hist.

        See Also
        --------
        sns_histplot : DataFrame-based histogram with KDE support.

        Examples
        --------
        >>> ax.hist(data, bins=20, density=True)
        >>> ax.hist(data, bins='auto', alpha=0.7, label='Group A')
        """
        method_name = "hist"

        axis_id = str(hash(self._axis_mpl))
        hist_id = id if id is not None else str(self.id)

        if align_bins:
            from .....plt.utils import histogram_bin_manager

            bins, range = histogram_bin_manager.register_histogram(
                axis_id, hist_id, x, bins, range
            )

        with self._no_tracking():
            hist_data = self._axis_mpl.hist(x, bins=bins, range=range, **kwargs)

        tracked_dict = {
            "args": (x,),
            "hist_result": (hist_data[0], hist_data[1]),
            "bins": bins,
            "range": range,
        }
        self._track(track, id, method_name, tracked_dict, kwargs)
        self._apply_scitex_postprocess(method_name, hist_data)

        return hist_data

    def stx_violin(
        self,
        data: Union[pd.DataFrame, List, ArrayLike],
        *,
        x: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        half: bool = False,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> "Axes":
        """Create a violin plot with SciTeX styling and tracking.

        Parameters
        ----------
        data : DataFrame, list, or array-like
            Data for the violin plot. Can be:
            - List of arrays (one per violin)
            - DataFrame with columns specified by x, y, hue
        x : str, optional
            Column name for x-axis grouping (DataFrame input).
        y : str, optional
            Column name for y-axis values (DataFrame input).
        hue : str, optional
            Column name for color grouping (DataFrame input).
        labels : list of str, optional
            Labels for each violin (list input).
        colors : list of str, optional
            Colors for each violin.
        half : bool, default False
            If True, draw half-violins (useful for paired comparisons).
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the violin function.

        Returns
        -------
        Axes
            The axes with the violin plot.

        See Also
        --------
        stx_violinplot : Alias for this method.
        sns_violinplot : DataFrame-based violin plot.
        stx_box : Boxplot alternative.

        Examples
        --------
        >>> ax.stx_violin([data1, data2], labels=['A', 'B'])
        >>> ax.stx_violin(df, x='group', y='value', hue='category')
        """
        method_name = "stx_violin"

        with self._no_tracking():
            if isinstance(data, list) and all(
                isinstance(item, (list, np.ndarray)) for item in data
            ):
                self._axis_mpl = self._get_ax_module().stx_violin(
                    self._axis_mpl,
                    values_list=data,
                    labels=labels,
                    colors=colors,
                    half=half,
                    **kwargs,
                )
            else:
                self._axis_mpl = self._get_ax_module().stx_violin(
                    self._axis_mpl,
                    data=data,
                    x=x,
                    y=y,
                    hue=hue,
                    half=half,
                    **kwargs,
                )

        tracked_dict = {
            "data": data,
            "x": x,
            "y": y,
            "hue": hue,
            "half": half,
            "labels": labels,
            "colors": colors,
        }
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_line(
        self,
        y: ArrayLike,
        *,
        x: Optional[ArrayLike] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", pd.DataFrame]:
        """Plot a simple line with SciTeX styling.

        Parameters
        ----------
        y : array-like
            Y values for the line.
        x : array-like, optional
            X values for the line. If None, uses integer indices.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the line plot function.

        Returns
        -------
        tuple
            (Axes, DataFrame) - The axes and a DataFrame with the plotted data.

        See Also
        --------
        stx_mean_std : Line with standard deviation shading.
        stx_shaded_line : Line with custom shaded region.
        sns_lineplot : DataFrame-based line plot.

        Examples
        --------
        >>> ax.stx_line(y_values)
        >>> ax.stx_line(y, x=x, label='Series A', color='blue')
        """
        method_name = "stx_line"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_line(
                self._axis_mpl, y, xx=x, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, kwargs)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    def stx_mean_std(
        self,
        data: ArrayLike,
        *,
        x: Optional[ArrayLike] = None,
        sd: float = 1.0,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", pd.DataFrame]:
        """Plot mean line with standard deviation shading.

        Parameters
        ----------
        data : array-like
            2D array where each row is an observation and columns are time points.
        x : array-like, optional
            X values. If None, uses integer indices.
        sd : float, default 1.0
            Number of standard deviations for the shaded region.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the plot function.

        Returns
        -------
        tuple
            (Axes, DataFrame) - The axes and a DataFrame with mean, upper, lower.

        See Also
        --------
        stx_mean_ci : Mean with confidence interval.
        stx_median_iqr : Median with interquartile range.
        stx_shaded_line : Custom shaded line.

        Examples
        --------
        >>> ax.stx_mean_std(data_2d, sd=2, label='Mean±2SD')
        """
        method_name = "stx_mean_std"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_mean_std(
                self._axis_mpl, data, xx=x, sd=sd, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    def stx_mean_ci(
        self,
        data: ArrayLike,
        *,
        x: Optional[ArrayLike] = None,
        ci: float = 95.0,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", pd.DataFrame]:
        """Plot mean line with confidence interval shading.

        Parameters
        ----------
        data : array-like
            2D array where each row is an observation and columns are time points.
        x : array-like, optional
            X values. If None, uses integer indices.
        ci : float, default 95.0
            Confidence interval percentage (e.g., 95 for 95% CI).
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the plot function.

        Returns
        -------
        tuple
            (Axes, DataFrame) - The axes and a DataFrame with mean, upper, lower.

        See Also
        --------
        stx_mean_std : Mean with standard deviation.
        stx_median_iqr : Median with interquartile range.

        Examples
        --------
        >>> ax.stx_mean_ci(data_2d, ci=99, label='Mean±99%CI')
        """
        method_name = "stx_mean_ci"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_mean_ci(
                self._axis_mpl, data, xx=x, perc=ci, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    def stx_median_iqr(
        self,
        data: ArrayLike,
        *,
        x: Optional[ArrayLike] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", pd.DataFrame]:
        """Plot median line with interquartile range shading.

        Parameters
        ----------
        data : array-like
            2D array where each row is an observation and columns are time points.
        x : array-like, optional
            X values. If None, uses integer indices.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the plot function.

        Returns
        -------
        tuple
            (Axes, DataFrame) - The axes and a DataFrame with median, Q1, Q3.

        See Also
        --------
        stx_mean_std : Mean with standard deviation.
        stx_mean_ci : Mean with confidence interval.

        Examples
        --------
        >>> ax.stx_median_iqr(data_2d, label='Median±IQR')
        """
        method_name = "stx_median_iqr"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_median_iqr(
                self._axis_mpl, data, xx=x, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    def stx_shaded_line(
        self,
        x: ArrayLike,
        y_lower: ArrayLike,
        y_middle: ArrayLike,
        y_upper: ArrayLike,
        *,
        color: Optional[Union[str, List[str]]] = None,
        label: Optional[Union[str, List[str]]] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", pd.DataFrame]:
        """Plot a line with shaded area between lower and upper bounds.

        Parameters
        ----------
        x : array-like
            X coordinates.
        y_lower : array-like
            Lower bound of the shaded region.
        y_middle : array-like
            Center line values.
        y_upper : array-like
            Upper bound of the shaded region.
        color : str or list of str, optional
            Color(s) for the line and shading.
        label : str or list of str, optional
            Label(s) for the legend.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the plot function.

        Returns
        -------
        tuple
            (Axes, DataFrame) - The axes and a DataFrame with the plotted data.

        See Also
        --------
        stx_mean_std : Mean with standard deviation.
        stx_fill_between : Simple fill between curves.

        Examples
        --------
        >>> ax.stx_shaded_line(x, lower, mean, upper, color='blue', label='Result')
        """
        method_name = "stx_shaded_line"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_shaded_line(
                self._axis_mpl,
                x,
                y_lower,
                y_middle,
                y_upper,
                color=color,
                label=label,
                **kwargs,
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df


# EOF
