#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _scientific.py - Scientific/specialized plot methods

"""Scientific and domain-specific plotting methods."""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from scitex.pd import to_xyz
from scitex.types import ArrayLike

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


class ScientificPlotMixin:
    """Mixin for scientific and domain-specific plotting methods.

    Provides specialized visualizations for:
    - Image display with colorbars
    - Kernel density estimation
    - Confusion matrices
    - Raster plots (spike trains)
    - ECDF plots
    - Joint distributions (scatter + marginal histograms)
    - Heatmaps with annotations
    """

    def stx_image(
        self,
        data: ArrayLike,
        *,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> "Axes":
        """Display a 2D array as an image with SciTeX styling.

        Parameters
        ----------
        data : array-like
            2D array to display as an image.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the image function.
            Common options: cmap, vmin, vmax, aspect, colorbar.

        Returns
        -------
        Axes
            The axes with the image displayed.

        See Also
        --------
        stx_imshow : Lower-level image display.
        stx_heatmap : Annotated heatmap.
        sns_heatmap : DataFrame-based heatmap.

        Examples
        --------
        >>> ax.stx_image(matrix, cmap='viridis', colorbar=True)
        """
        method_name = "stx_image"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_image(
                self._axis_mpl, data, **kwargs
            )

        tracked_dict = {"image_df": pd.DataFrame(data)}
        if kwargs.get("xyz", False):
            tracked_dict["image_df"] = to_xyz(tracked_dict["image_df"])
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_kde(
        self,
        data: ArrayLike,
        *,
        cumulative: bool = False,
        fill: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> "Axes":
        """Plot a kernel density estimate of the data.

        Parameters
        ----------
        data : array-like
            1D array of values for density estimation.
        cumulative : bool, default False
            If True, plot cumulative distribution instead of density.
        fill : bool, default False
            If True, fill the area under the curve.
        xlim : tuple of float, optional
            Range for the x-axis. If None, uses data range.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the plot function.
            Common options: color, linewidth, linestyle, label.

        Returns
        -------
        Axes
            The axes with the KDE plot.

        See Also
        --------
        sns_kdeplot : DataFrame-based KDE plot.
        stx_ecdf : Empirical cumulative distribution function.
        hist : Histogram alternative.

        Examples
        --------
        >>> ax.stx_kde(samples, fill=True, alpha=0.3)
        >>> ax.stx_kde(data, cumulative=True, label='CDF')
        """
        method_name = "stx_kde"

        n_samples = (~np.isnan(data)).sum()
        if kwargs.get("label"):
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        if xlim is None:
            xlim = (np.nanmin(data), np.nanmax(data))

        xx = np.linspace(xlim[0], xlim[1], int(1e3))
        density = gaussian_kde(data)(xx)
        density /= density.sum()

        if cumulative:
            density = np.cumsum(density)

        with self._no_tracking():
            from scitex.plt.utils import mm_to_pt

            if "linewidth" not in kwargs and "lw" not in kwargs:
                kwargs["linewidth"] = mm_to_pt(0.2)
            if "color" not in kwargs and "c" not in kwargs:
                kwargs["color"] = "black"
            if "linestyle" not in kwargs and "ls" not in kwargs:
                kwargs["linestyle"] = "--"

            if fill:
                self._axis_mpl.fill_between(xx, density, **kwargs)
            else:
                self._axis_mpl.plot(xx, density, **kwargs)

        tracked_dict = {"x": xx, "kde": density, "n": n_samples}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_conf_mat(
        self,
        data: ArrayLike,
        *,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        cbar: bool = True,
        cbar_kw: Optional[Dict[str, Any]] = None,
        label_rotation_xy: Tuple[float, float] = (15, 15),
        x_extend_ratio: float = 1.0,
        y_extend_ratio: float = 1.0,
        calc_bacc: bool = False,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", Optional[float]]:
        """Plot a confusion matrix with optional balanced accuracy calculation.

        Parameters
        ----------
        data : array-like
            2D confusion matrix array.
        x_labels : list of str, optional
            Labels for x-axis (predicted classes).
        y_labels : list of str, optional
            Labels for y-axis (true classes).
        title : str, default 'Confusion Matrix'
            Title for the plot.
        cmap : str, default 'Blues'
            Colormap for the heatmap.
        cbar : bool, default True
            Whether to show the colorbar.
        cbar_kw : dict, optional
            Additional keyword arguments for the colorbar.
        label_rotation_xy : tuple of float, default (15, 15)
            Rotation angles for (x, y) axis labels.
        x_extend_ratio : float, default 1.0
            Ratio to extend x-axis limits.
        y_extend_ratio : float, default 1.0
            Ratio to extend y-axis limits.
        calc_bacc : bool, default False
            Whether to calculate and return balanced accuracy.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the heatmap function.

        Returns
        -------
        tuple
            (Axes, balanced_accuracy) - balanced_accuracy is None if calc_bacc=False.

        Examples
        --------
        >>> ax.stx_conf_mat(cm, x_labels=['A', 'B'], y_labels=['A', 'B'])
        >>> ax, bacc = ax.stx_conf_mat(cm, calc_bacc=True)
        """
        method_name = "stx_conf_mat"

        if cbar_kw is None:
            cbar_kw = {}

        with self._no_tracking():
            self._axis_mpl, bacc_val = self._get_ax_module().stx_conf_mat(
                self._axis_mpl,
                data,
                x_labels=x_labels,
                y_labels=y_labels,
                title=title,
                cmap=cmap,
                cbar=cbar,
                cbar_kw=cbar_kw,
                label_rotation_xy=label_rotation_xy,
                x_extend_ratio=x_extend_ratio,
                y_extend_ratio=y_extend_ratio,
                calc_bacc=calc_bacc,
                **kwargs,
            )

        tracked_dict = {
            "args": [data],
            "balanced_accuracy": bacc_val,
            "x_labels": x_labels,
            "y_labels": y_labels,
        }
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, bacc_val

    def stx_raster(
        self,
        spike_times: List[ArrayLike],
        *,
        time: Optional[ArrayLike] = None,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", pd.DataFrame]:
        """Plot a raster plot (spike train visualization).

        Parameters
        ----------
        spike_times : list of array-like
            List of arrays, each containing spike times for one unit/neuron.
        time : array-like, optional
            Time axis reference. If None, uses spike time range.
        labels : list of str, optional
            Labels for each unit/row.
        colors : list of str, optional
            Colors for each unit/row.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the raster function.

        Returns
        -------
        tuple
            (Axes, DataFrame) - The axes and digitized raster data.

        Examples
        --------
        >>> ax.stx_raster([spikes_unit1, spikes_unit2], labels=['Unit 1', 'Unit 2'])
        """
        method_name = "stx_raster"

        with self._no_tracking():
            self._axis_mpl, raster_digit_df = self._get_ax_module().stx_raster(
                self._axis_mpl, spike_times, time=time
            )

        tracked_dict = {"raster_digit_df": raster_digit_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, raster_digit_df

    def stx_ecdf(
        self,
        data: ArrayLike,
        *,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", pd.DataFrame]:
        """Plot an empirical cumulative distribution function (ECDF).

        Parameters
        ----------
        data : array-like
            1D array of values.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the ECDF function.
            Common options: color, linewidth, label.

        Returns
        -------
        tuple
            (Axes, DataFrame) - The axes and ECDF data (x, y columns).

        See Also
        --------
        stx_kde : Kernel density estimate (continuous).
        hist : Histogram (discrete bins).

        Examples
        --------
        >>> ax.stx_ecdf(samples, label='Distribution A')
        """
        method_name = "stx_ecdf"

        with self._no_tracking():
            self._axis_mpl, ecdf_df = self._get_ax_module().stx_ecdf(
                self._axis_mpl, data, **kwargs
            )

        tracked_dict = {"ecdf_df": ecdf_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, ecdf_df

    def stx_joyplot(
        self,
        data: ArrayLike,
        *,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> "Axes":
        """Plot a joyplot (ridgeline plot) for distribution comparison.

        Parameters
        ----------
        data : array-like
            2D array where each row is a distribution to plot.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the joyplot function.

        Returns
        -------
        Axes
            The axes with the joyplot.

        Examples
        --------
        >>> ax.stx_joyplot(distributions_2d, overlap=0.5)
        """
        method_name = "stx_joyplot"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_joyplot(
                self._axis_mpl, data, **kwargs
            )

        tracked_dict = {"joyplot_data": data}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_scatter_hist(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        hist_bins: int = 20,
        scatter_alpha: float = 0.6,
        scatter_size: float = 20,
        scatter_color: str = "blue",
        hist_color_x: str = "blue",
        hist_color_y: str = "red",
        hist_alpha: float = 0.5,
        scatter_ratio: float = 0.8,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", "Axes", "Axes", Dict]:
        """Plot a scatter plot with marginal histograms.

        Parameters
        ----------
        x : array-like
            X coordinates of the scatter points.
        y : array-like
            Y coordinates of the scatter points.
        hist_bins : int, default 20
            Number of bins for the marginal histograms.
        scatter_alpha : float, default 0.6
            Transparency of scatter points.
        scatter_size : float, default 20
            Size of scatter points.
        scatter_color : str, default 'blue'
            Color of scatter points.
        hist_color_x : str, default 'blue'
            Color of x-marginal histogram.
        hist_color_y : str, default 'red'
            Color of y-marginal histogram.
        hist_alpha : float, default 0.5
            Transparency of histograms.
        scatter_ratio : float, default 0.8
            Ratio of scatter plot area to total area.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the scatter function.

        Returns
        -------
        tuple
            (main_ax, hist_x_ax, hist_y_ax, hist_data) - Axes and histogram data.

        See Also
        --------
        stx_scatter : Simple scatter plot.
        sns_jointplot : Seaborn joint plot.

        Examples
        --------
        >>> ax, ax_hx, ax_hy, data = ax.stx_scatter_hist(x, y, hist_bins=30)
        """
        method_name = "stx_scatter_hist"

        with self._no_tracking():
            self._axis_mpl, ax_histx, ax_histy, hist_data = (
                self._get_ax_module().stx_scatter_hist(
                    self._axis_mpl,
                    x,
                    y,
                    hist_bins=hist_bins,
                    scatter_alpha=scatter_alpha,
                    scatter_size=scatter_size,
                    scatter_color=scatter_color,
                    hist_color_x=hist_color_x,
                    hist_color_y=hist_color_y,
                    hist_alpha=hist_alpha,
                    scatter_ratio=scatter_ratio,
                    **kwargs,
                )
            )

        tracked_dict = {
            "x": x,
            "y": y,
            "hist_x": hist_data["hist_x"],
            "hist_y": hist_data["hist_y"],
            "bin_edges_x": hist_data["bin_edges_x"],
            "bin_edges_y": hist_data["bin_edges_y"],
        }
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, ax_histx, ax_histy, hist_data

    def stx_heatmap(
        self,
        data: ArrayLike,
        *,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        cmap: str = "viridis",
        cbar_label: str = "ColorBar Label",
        value_format: str = "{x:.1f}",
        show_annot: bool = True,
        annot_color_lighter: str = "white",
        annot_color_darker: str = "black",
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["Axes", matplotlib.image.AxesImage, matplotlib.colorbar.Colorbar]:
        """Plot an annotated heatmap.

        Parameters
        ----------
        data : array-like
            2D array of values to display.
        x_labels : list of str, optional
            Labels for x-axis (columns).
        y_labels : list of str, optional
            Labels for y-axis (rows).
        cmap : str, default 'viridis'
            Colormap name.
        cbar_label : str, default 'ColorBar Label'
            Label for the colorbar.
        value_format : str, default '{x:.1f}'
            Format string for cell annotations.
        show_annot : bool, default True
            Whether to show value annotations in cells.
        annot_color_lighter : str, default 'white'
            Annotation color for dark backgrounds.
        annot_color_darker : str, default 'black'
            Annotation color for light backgrounds.
        track : bool, default True
            Enable data tracking for reproducibility.
        id : str, optional
            Unique identifier for this plot element.
        **kwargs
            Additional arguments passed to the heatmap function.

        Returns
        -------
        tuple
            (Axes, AxesImage, Colorbar) - The axes, image, and colorbar objects.

        See Also
        --------
        sns_heatmap : DataFrame-based heatmap.
        stx_conf_mat : Confusion matrix heatmap.
        stx_image : Simple image display.

        Examples
        --------
        >>> ax, im, cbar = ax.stx_heatmap(matrix, x_labels=['A', 'B'], cmap='coolwarm')
        """
        method_name = "stx_heatmap"

        with self._no_tracking():
            ax, im, cbar = self._get_ax_module().stx_heatmap(
                self._axis_mpl,
                data,
                x_labels=x_labels,
                y_labels=y_labels,
                cmap=cmap,
                cbar_label=cbar_label,
                value_format=value_format,
                show_annot=show_annot,
                annot_color_lighter=annot_color_lighter,
                annot_color_darker=annot_color_darker,
                **kwargs,
            )

        tracked_dict = {
            "data": data,
            "x_labels": x_labels,
            "y_labels": y_labels,
        }
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return ax, im, cbar


# EOF
