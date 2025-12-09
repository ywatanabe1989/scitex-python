#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_MatplotlibPlotMixin.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from scitex.pd import to_xyz
from scitex.types import ArrayLike
from scitex.plt.utils import mm_to_pt


# ============================================================================
# Constants for default styling (same as styles/_plot_defaults.py)
# ============================================================================
DEFAULT_LINE_WIDTH_MM = 0.2
DEFAULT_MARKER_SIZE_MM = 0.8
DEFAULT_FILL_ALPHA = 0.3


class MatplotlibPlotMixin:
    """Mixin class for basic plotting operations."""

    def _get_ax_module(self):
        """Lazy import ax module to avoid circular imports."""
        from ....plt import ax as ax_module

        return ax_module

    def _apply_scitex_postprocess(
        self, method_name, result=None, kwargs=None, args=None
    ):
        """Apply scitex post-processing styling after plotting.

        This ensures all scitex wrapper methods get the same styling
        as matplotlib methods going through __getattr__ (tick locator, spines, etc.).
        """
        from scitex.plt.styles import apply_plot_postprocess

        apply_plot_postprocess(method_name, result, self._axis_mpl, kwargs or {}, args)

    def stx_image(
        self,
        arr_2d: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_image"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_image(
                self._axis_mpl, arr_2d, **kwargs
            )

        # Tracking
        tracked_dict = {"image_df": pd.DataFrame(arr_2d)}
        if kwargs.get("xyz", False):
            tracked_dict["image_df"] = to_xyz(tracked_dict["image_df"])
        self._track(
            track,
            id,
            method_name,
            tracked_dict,
            None,
        )

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_kde(
        self,
        values_1d: ArrayLike,
        cumulative=False,
        fill=False,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_kde"

        # Sample count as label
        n_samples = (~np.isnan(values_1d)).sum()
        if kwargs.get("label"):
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        # Xlim (kwargs["xlim"] is not accepted in downstream plotters)
        xlim = kwargs.pop("xlim", None)
        if not xlim:
            xlim = (np.nanmin(values_1d), np.nanmax(values_1d))

        # X
        xx = np.linspace(xlim[0], xlim[1], int(1e3))

        # Y
        density = gaussian_kde(values_1d)(xx)
        density /= density.sum()

        # Cumulative
        if cumulative:
            density = np.cumsum(density)

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            # Get line width from kwargs or use default (0.2mm for KDE)
            from scitex.plt.utils import mm_to_pt

            if "linewidth" not in kwargs and "lw" not in kwargs:
                kwargs["linewidth"] = mm_to_pt(0.2)  # Default 0.2mm for KDE

            # Set default color to black (customizable via color kwarg)
            if "color" not in kwargs and "c" not in kwargs:
                kwargs["color"] = "black"

            # Set default linestyle to dashed (customizable via linestyle kwarg)
            if "linestyle" not in kwargs and "ls" not in kwargs:
                kwargs["linestyle"] = "--"

            # Filled Line
            if fill:
                self._axis_mpl.fill_between(
                    xx,
                    density,
                    **kwargs,
                )
            # Simple Line
            else:
                self._axis_mpl.plot(xx, density, **kwargs)

        # Tracking
        tracked_dict = {
            "x": xx,
            "kde": density,
            "n": n_samples,
        }
        self._track(
            track,
            id,
            method_name,
            tracked_dict,
            None,
        )

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_conf_mat(
        self,
        conf_mat_2d: ArrayLike,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        cbar: bool = True,
        cbar_kw: Dict[str, Any] = {},
        label_rotation_xy: Tuple[float, float] = (15, 15),
        x_extend_ratio: float = 1.0,
        y_extend_ratio: float = 1.0,
        calc_bacc: bool = False,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_conf_mat"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, bacc_val = self._get_ax_module().stx_conf_mat(
                self._axis_mpl,
                conf_mat_2d,
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

        tracked_dict = {"balanced_accuracy": bacc_val}
        # Tracking
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, bacc_val

    # @wraps removed to avoid circular import
    def stx_rectangle(
        self,
        xx: float,
        yy: float,
        width: float,
        height: float,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_rectangle"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_rectangle(
                self._axis_mpl, xx, yy, width, height, **kwargs
            )

        # Tracking - use "x", "y" to match formatter expected keys
        tracked_dict = {"x": xx, "y": yy, "width": width, "height": height}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    # @wraps removed to avoid circular import
    def stx_fillv(
        self,
        starts_1d: ArrayLike,
        ends_1d: ArrayLike,
        color: str = "red",
        alpha: float = 0.2,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_fillv"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_fillv(
                self._axis_mpl, starts_1d, ends_1d, color=color, alpha=alpha
            )

        # Tracking
        tracked_dict = {"starts": starts_1d, "ends": ends_1d}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_box(
        self,
        values_list: ArrayLike,
        colors: Optional[List] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        # Method Name for downstream csv exporting
        method_name = "stx_box"

        # Copy data
        _data = values_list.copy()

        # Sample count per group as label (show range if variable)
        if kwargs.get("label"):
            n_per_group = [len(g) for g in values_list]
            n_min, n_max = min(n_per_group), max(n_per_group)
            n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
            kwargs["label"] = kwargs["label"] + f" ($n$={n_str})"

        # Enable patch_artist for styling (fill colors, edges)
        if "patch_artist" not in kwargs:
            kwargs["patch_artist"] = True

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            result = self._axis_mpl.boxplot(values_list, **kwargs)

        # Tracking - calculate sample size per group
        n_per_group = [len(g) for g in values_list]
        tracked_dict = {
            "data": _data,
            "n": n_per_group,
        }
        self._track(track, id, method_name, tracked_dict, None)

        # Apply style_boxplot automatically for publication quality
        # Uses scitex palette by default, or custom colors if provided
        from scitex.plt.ax import style_boxplot

        style_boxplot(result, colors=colors)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def hist(
        self,
        x: ArrayLike,
        bins: Union[int, str, ArrayLike] = 10,
        range: Optional[Tuple[float, float]] = None,
        align_bins: bool = True,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Plot a histogram.

        This is an override of the standard matplotlib hist function to ensure
        that histogram bin data is properly tracked for CSV export and bins are
        aligned for histograms on the same axis.

        Args:
            x: Input data
            bins: Bin specification (count, edges, or algorithm)
            range: Optional histogram range (min, max)
            align_bins: Whether to align bins with other histograms on this axis
            track: Whether to track this operation
            id: Identifier for tracking
            **kwargs: Additional keywords passed to matplotlib hist

        Returns:
            Histogram output
        """
        # Method Name for downstream csv exporting
        method_name = "hist"

        # Get the axis ID for bin alignment
        axis_id = str(hash(self._axis_mpl))
        hist_id = id if id is not None else str(self.id)

        # Align bins if requested and not the first histogram on this axis
        if align_bins:
            from ....plt.utils import histogram_bin_manager

            bins, range = histogram_bin_manager.register_histogram(
                axis_id, hist_id, x, bins, range
            )

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            hist_data = self._axis_mpl.hist(x, bins=bins, range=range, **kwargs)

        # Save histogram result for CSV export
        # hist_data[0] = counts, hist_data[1] = bin_edges
        tracked_dict = {
            "args": (x,),
            "hist_result": (hist_data[0], hist_data[1]),
            "bins": bins,
            "range": range,
        }

        self._track(track, id, method_name, tracked_dict, kwargs)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, hist_data)

        return hist_data

    # @wraps removed to avoid circular import
    def stx_raster(
        self,
        spike_times_list: List[ArrayLike],
        time: Optional[ArrayLike] = None,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_raster"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, raster_digit_df = self._get_ax_module().stx_raster(
                self._axis_mpl, spike_times_list, time=time
            )

        # Tracking
        tracked_dict = {"raster_digit_df": raster_digit_df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, raster_digit_df

    # @wraps removed to avoid circular import
    def stx_ecdf(
        self,
        values_1d: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_ecdf"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, ecdf_df = self._get_ax_module().stx_ecdf(
                self._axis_mpl, values_1d, **kwargs
            )

        # Tracking
        tracked_dict = {"ecdf_df": ecdf_df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, ecdf_df

    # @wraps removed to avoid circular import
    def stx_joyplot(
        self,
        arrays: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Method Name for downstream csv exporting
        method_name = "stx_joyplot"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_joyplot(
                self._axis_mpl, arrays, **kwargs
            )

        # Tracking - use "joyplot_data" to match formatter expected key
        tracked_dict = {"joyplot_data": arrays}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    # @wraps removed to avoid circular import
    def stx_scatter_hist(
        self,
        x: ArrayLike,
        y: ArrayLike,
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
    ) -> None:
        """Plot a scatter plot with marginal histograms."""
        # Method Name for downstream csv exporting
        method_name = "stx_scatter_hist"

        # Plotting with pure matplotlib methods under non-tracking context
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

        # Tracking
        tracked_dict = {
            "x": x,
            "y": y,
            "hist_x": hist_data["hist_x"],
            "hist_y": hist_data["hist_y"],
            "bin_edges_x": hist_data["bin_edges_x"],
            "bin_edges_y": hist_data["bin_edges_y"],
        }
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, ax_histx, ax_histy, hist_data

    # @wraps removed to avoid circular import
    def stx_heatmap(
        self,
        values_2d: ArrayLike,
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
    ) -> Tuple[matplotlib.image.AxesImage, matplotlib.colorbar.Colorbar]:
        """Plot a heatmap on the axes."""
        # Method Name for downstream csv exporting
        method_name = "stx_heatmap"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            ax, im, cbar = self._get_ax_module().stx_heatmap(
                self._axis_mpl,
                values_2d,
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

        # Tracking
        tracked_dict = {
            "data": values_2d,
            "x_labels": x_labels,
            "y_labels": y_labels,
        }
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return ax, im, cbar

    # @wraps removed to avoid circular import
    def stx_violin(
        self,
        values_list: Union[pd.DataFrame, List, ArrayLike],
        x=None,
        y=None,
        hue=None,
        labels=None,
        colors=None,
        half=False,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot a violin plot."""
        # Method Name for downstream csv exporting
        method_name = "stx_violin"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            # Handle the list-style input case
            if isinstance(values_list, list) and all(
                isinstance(item, (list, np.ndarray)) for item in values_list
            ):
                self._axis_mpl = self._get_ax_module().stx_violin(
                    self._axis_mpl,
                    values_list=values_list,
                    labels=labels,
                    colors=colors,
                    half=half,
                    **kwargs,
                )
            # Handle DataFrame or other inputs
            else:
                self._axis_mpl = self._get_ax_module().stx_violin(
                    self._axis_mpl,
                    data=values_list,
                    x=x,
                    y=y,
                    hue=hue,
                    half=half,
                    **kwargs,
                )

        # Tracking
        tracked_dict = {
            "data": values_list,
            "x": x,
            "y": y,
            "hue": hue,
            "half": half,
            "labels": labels,
            "colors": colors,
        }
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    # def plot_area(
    #     self,
    #     x: ArrayLike,
    #     y: ArrayLike,
    #     stacked: bool = False,
    #     fill: bool = True,
    #     alpha: float = 0.5,
    #     track: bool = True,
    #     id: Optional[str] = None,
    #     **kwargs,
    # ) -> None:
    #     """Plot an area plot."""
    #     # Method Name for downstream csv exporting
    #     method_name = "plot_area"

    #     # Plotting with pure matplotlib methods under non-tracking context
    #     with self._no_tracking():
    #         self._axis_mpl = self._get_ax_module().plot_area(
    #             self._axis_mpl,
    #             x,
    #             y,
    #             stacked=stacked,
    #             fill=fill,
    #             alpha=alpha,
    #             **kwargs,
    #         )

    #     # Tracking
    #     tracked_dict = {"x": x, "y": y}
    #     self._track(track, id, method_name, tracked_dict, None)

    #     return self._axis_mpl

    # def plot_radar(
    #     self,
    #     data: ArrayLike,
    #     categories: List[str],
    #     groups: Optional[List[str]] = None,
    #     fill: bool = True,
    #     alpha: float = 0.2,
    #     grid_step: int = 5,
    #     track: bool = True,
    #     id: Optional[str] = None,
    #     **kwargs,
    # ) -> None:
    #     """Plot a radar/spider chart."""
    #     # Method Name for downstream csv exporting
    #     method_name = "plot_radar"

    #     # Convert data to DataFrame if not already
    #     if not isinstance(data, pd.DataFrame):
    #         if groups is not None:
    #             data = pd.DataFrame(data, columns=categories, index=groups)
    #         else:
    #             data = pd.DataFrame(data, columns=categories)

    #     # Plotting with pure matplotlib methods under non-tracking context
    #     with self._no_tracking():
    #         self._axis_mpl = self._get_ax_module().plot_radar(
    #             self._axis_mpl,
    #             data,
    #             categories=categories,
    #             fill=fill,
    #             alpha=alpha,
    #             grid_step=grid_step,
    #             **kwargs,
    #         )

    #     # Tracking
    #     tracked_dict = {"radar_data": data}
    #     self._track(track, id, method_name, tracked_dict, None)

    #     return self._axis_mpl

    # def plot_bubble(
    #     self,
    #     x: ArrayLike,
    #     y: ArrayLike,
    #     size: ArrayLike,
    #     color: Optional[ArrayLike] = None,
    #     size_scale: float = 1000.0,
    #     alpha: float = 0.6,
    #     colormap: str = "viridis",
    #     show_colorbar: bool = True,
    #     colorbar_label: str = "",
    #     track: bool = True,
    #     id: Optional[str] = None,
    #     **kwargs,
    # ) -> None:
    #     """Plot a bubble chart."""
    #     # Method Name for downstream csv exporting
    #     method_name = "plot_bubble"

    #     # Plotting with pure matplotlib methods under non-tracking context
    #     with self._no_tracking():
    #         self._axis_mpl = self._get_ax_module().plot_bubble(
    #             self._axis_mpl,
    #             x,
    #             y,
    #             size,
    #             color=color,
    #             size_scale=size_scale,
    #             alpha=alpha,
    #             colormap=colormap,
    #             show_colorbar=show_colorbar,
    #             colorbar_label=colorbar_label,
    #             **kwargs,
    #         )

    #     # Tracking
    #     tracked_dict = {"x": x, "y": y, "size": size}
    #     if color is not None:
    #         tracked_dict["color"] = color

    #     self._track(track, id, method_name, tracked_dict, None)

    #     return self._axis_mpl

    # def plot_ridgeline(
    #     self,
    #     data: ArrayLike,
    #     labels: Optional[List[str]] = None,
    #     overlap: float = 0.8,
    #     fill: bool = True,
    #     alpha: float = 0.6,
    #     colormap: str = "viridis",
    #     bandwidth: Optional[float] = None,
    #     track: bool = True,
    #     id: Optional[str] = None,
    #     **kwargs,
    # ) -> None:
    #     """Plot a ridgeline plot (similar to joyplot but with KDE)."""
    #     # Method Name for downstream csv exporting
    #     method_name = "plot_ridgeline"

    #     # Ensure data is in correct format
    #     if isinstance(data, pd.DataFrame):
    #         _data = [data[col].dropna().values for col in data.columns]
    #         if labels is None:
    #             labels = list(data.columns)
    #     elif isinstance(data, list):
    #         _data = data
    #     else:
    #         _data = [data]

    #     # Plotting with pure matplotlib methods under non-tracking context
    #     with self._no_tracking():
    #         self._axis_mpl, ridge_data = self._get_ax_module().plot_ridgeline(
    #             self._axis_mpl,
    #             _data,
    #             labels=labels,
    #             overlap=overlap,
    #             fill=fill,
    #             alpha=alpha,
    #             colormap=colormap,
    #             bandwidth=bandwidth,
    #             **kwargs,
    #         )

    #     # Tracking
    #     tracked_dict = {
    #         "ridgeline_data": _data,
    #         "kde_x": ridge_data["kde_x"],
    #         "kde_y": ridge_data["kde_y"],
    #     }
    #     if labels is not None:
    #         tracked_dict["labels"] = labels
    #     self._track(track, id, method_name, tracked_dict, None)

    #     return self._axis_mpl, ridge_data

    # def plot_parallel_coordinates(
    #     self,
    #     data: pd.DataFrame,
    #     class_column: Optional[str] = None,
    #     colormap: str = "viridis",
    #     alpha: float = 0.5,
    #     track: bool = True,
    #     id: Optional[str] = None,
    #     **kwargs,
    # ) -> None:
    #     """Plot parallel coordinates."""
    #     # Method Name for downstream csv exporting
    #     method_name = "plot_parallel_coordinates"

    #     # Plotting with pure matplotlib methods under non-tracking context
    #     with self._no_tracking():
    #         self._axis_mpl = self._get_ax_module().plot_parallel_coordinates(
    #             self._axis_mpl,
    #             data,
    #             class_column=class_column,
    #             colormap=colormap,
    #             alpha=alpha,
    #             **kwargs,
    #         )

    #     # Tracking
    #     tracked_dict = {"parallel_data": data}
    #     self._track(track, id, method_name, tracked_dict, None)

    #     return self._axis_mpl

    # @wraps removed to avoid circular import
    def stx_line(
        self,
        values_1d: ArrayLike,
        xx: Optional[ArrayLike] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot a simple line."""
        # Method Name for downstream csv exporting
        method_name = "stx_line"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_line(
                self._axis_mpl, values_1d, xx=xx, **kwargs
            )

        # Tracking
        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    # @wraps removed to avoid circular import
    def stx_mean_std(
        self,
        values_2d: ArrayLike,
        xx: Optional[ArrayLike] = None,
        sd: float = 1,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot mean line with standard deviation shading."""
        # Method Name for downstream csv exporting
        method_name = "stx_mean_std"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_mean_std(
                self._axis_mpl, values_2d, xx=xx, sd=sd, **kwargs
            )

        # Tracking
        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    # @wraps removed to avoid circular import
    def stx_mean_ci(
        self,
        values_2d: ArrayLike,
        xx: Optional[ArrayLike] = None,
        perc: float = 95,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot mean line with confidence interval shading."""
        # Method Name for downstream csv exporting
        method_name = "stx_mean_ci"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_mean_ci(
                self._axis_mpl, values_2d, xx=xx, perc=perc, **kwargs
            )

        # Tracking
        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    # @wraps removed to avoid circular import
    def stx_median_iqr(
        self,
        values_2d: ArrayLike,
        xx: Optional[ArrayLike] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot median line with interquartile range shading."""
        # Method Name for downstream csv exporting
        method_name = "stx_median_iqr"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_median_iqr(
                self._axis_mpl, values_2d, xx=xx, **kwargs
            )

        # Tracking
        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    # @wraps removed to avoid circular import
    def stx_shaded_line(
        self,
        xs: ArrayLike,
        ys_lower: ArrayLike,
        ys_middle: ArrayLike,
        ys_upper: ArrayLike,
        color: str or Optional[Union[str, List[str]]] = None,
        label: str or Optional[Union[str, List[str]]] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot a line with shaded area between lower and upper bounds."""
        # Method Name for downstream csv exporting
        method_name = "stx_shaded_line"

        # Plotting with pure matplotlib methods under non-tracking context
        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_shaded_line(
                self._axis_mpl,
                xs,
                ys_lower,
                ys_middle,
                ys_upper,
                color=color,
                label=label,
                **kwargs,
            )

        # Tracking
        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    # =========================================================================
    # stx_ aliases for standard matplotlib methods
    # These provide a consistent stx_ prefix for all scitex wrapper methods
    # =========================================================================

    def stx_bar(
        self, x, height, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Bar plot with scitex styling and tracking.

        Parameters
        ----------
        x : array-like
            The x coordinates of the bars
        height : array-like
            The heights of the bars
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib bar
        """
        method_name = "stx_bar"

        # Add sample size to label if provided
        if kwargs.get("label"):
            n_samples = len(x)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.bar(x, height, **kwargs)

        # Track bar data
        tracked_dict = {"bar_df": pd.DataFrame({"x": x, "height": height})}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply style_barplot automatically for publication quality
        from scitex.plt.ax import style_barplot

        style_barplot(result)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_barh(
        self, y, width, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Horizontal bar plot with scitex styling and tracking.

        Parameters
        ----------
        y : array-like
            The y coordinates of the bars
        width : array-like
            The widths of the bars
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib barh
        """
        method_name = "stx_barh"

        # Add sample size to label if provided
        if kwargs.get("label"):
            n_samples = len(y)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.barh(y, width, **kwargs)

        # Track bar data
        tracked_dict = {"barh_df": pd.DataFrame({"y": y, "width": width})}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_scatter(self, x, y, track: bool = True, id: Optional[str] = None, **kwargs):
        """Scatter plot with scitex styling and tracking.

        Parameters
        ----------
        x : array-like
            The x coordinates of the points
        y : array-like
            The y coordinates of the points
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib scatter
        """
        method_name = "stx_scatter"

        # Add sample size to label if provided
        if kwargs.get("label"):
            n_samples = len(x)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.scatter(x, y, **kwargs)

        # Track scatter data
        tracked_dict = {"scatter_df": pd.DataFrame({"x": x, "y": y})}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply style_scatter automatically for publication quality
        from scitex.plt.ax import style_scatter

        style_scatter(result)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_errorbar(
        self,
        x,
        y,
        yerr=None,
        xerr=None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ):
        """Error bar plot with scitex styling and tracking.

        Parameters
        ----------
        x : array-like
            The x coordinates of the data points
        y : array-like
            The y coordinates of the data points
        yerr : array-like, optional
            The y error values
        xerr : array-like, optional
            The x error values
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib errorbar
        """
        method_name = "stx_errorbar"

        # Add sample size to label if provided
        if kwargs.get("label"):
            n_samples = len(x)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.errorbar(x, y, yerr=yerr, xerr=xerr, **kwargs)

        # Track errorbar data
        df_dict = {"x": x, "y": y}
        if yerr is not None:
            df_dict["yerr"] = yerr
        if xerr is not None:
            df_dict["xerr"] = xerr
        tracked_dict = {"errorbar_df": pd.DataFrame(df_dict)}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply style_errorbar automatically for publication quality
        from scitex.plt.ax import style_errorbar

        style_errorbar(result)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_fill_between(
        self, x, y1, y2=0, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Fill between plot with scitex styling and tracking.

        Parameters
        ----------
        x : array-like
            The x coordinates
        y1 : array-like
            The first y boundary
        y2 : array-like or scalar, optional
            The second y boundary (default 0)
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib fill_between
        """
        method_name = "stx_fill_between"

        with self._no_tracking():
            result = self._axis_mpl.fill_between(x, y1, y2, **kwargs)

        # Track fill_between data
        tracked_dict = {
            "fill_between_df": pd.DataFrame(
                {
                    "x": x,
                    "y1": y1,
                    "y2": y2 if hasattr(y2, "__len__") else [y2] * len(x),
                }
            )
        }
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_contour(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Contour plot with scitex styling and tracking.

        Parameters
        ----------
        *args
            Positional arguments passed to matplotlib contour (X, Y, Z)
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib contour
        """
        method_name = "stx_contour"

        with self._no_tracking():
            result = self._axis_mpl.contour(*args, **kwargs)

        # Track contour data
        if len(args) >= 3:
            X, Y, Z = args[0], args[1], args[2]
            tracked_dict = {
                "contour_df": pd.DataFrame(
                    {"X": np.ravel(X), "Y": np.ravel(Y), "Z": np.ravel(Z)}
                )
            }
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_imshow(self, data, track: bool = True, id: Optional[str] = None, **kwargs):
        """Image display with scitex styling and tracking.

        Parameters
        ----------
        data : array-like
            2D array of image data
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib imshow
        """
        method_name = "stx_imshow"

        with self._no_tracking():
            result = self._axis_mpl.imshow(data, **kwargs)

        # Track image data
        if hasattr(data, "shape") and len(data.shape) == 2:
            n_rows, n_cols = data.shape
            df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])
        else:
            df = pd.DataFrame(data)
        tracked_dict = {"imshow_df": df}
        self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_boxplot(
        self,
        data,
        colors: Optional[List] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ):
        """Boxplot with scitex styling and tracking (alias for stx_box).

        Parameters
        ----------
        data : list of array-like
            List of data arrays for each box
        colors : list, optional
            Colors for each box
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to matplotlib boxplot
        """
        return self.stx_box(data, colors=colors, track=track, id=id, **kwargs)

    def stx_violinplot(
        self,
        data,
        colors: Optional[List] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ):
        """Violinplot with scitex styling and tracking (alias for stx_violin).

        Parameters
        ----------
        data : list of array-like or DataFrame
            Data for violin plot
        colors : list, optional
            Colors for each violin
        track : bool
            Whether to track data for CSV export
        id : str, optional
            Identifier for tracking
        **kwargs
            Additional arguments passed to stx_violin
        """
        return self.stx_violin(data, colors=colors, track=track, id=id, **kwargs)

    # Standard matplotlib plot methods with plot_ prefix
    def plot_bar(self, *args, track: bool = True, id: Optional[str] = None, **kwargs):
        """Wrapper for matplotlib bar plot with tracking support."""
        method_name = "plot_bar"

        with self._no_tracking():
            result = self._axis_mpl.bar(*args, **kwargs)

        # Track bar data
        if len(args) >= 2:
            tracked_dict = {"bar_df": pd.DataFrame({"x": args[0], "height": args[1]})}
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_barh(self, *args, track: bool = True, id: Optional[str] = None, **kwargs):
        """Wrapper for matplotlib horizontal bar plot with tracking support."""
        method_name = "plot_barh"

        with self._no_tracking():
            result = self._axis_mpl.barh(*args, **kwargs)

        # Track bar data
        if len(args) >= 2:
            tracked_dict = {"barh_df": pd.DataFrame({"y": args[0], "width": args[1]})}
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_scatter(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib scatter plot with tracking support."""
        method_name = "plot_scatter"

        # Add sample size to label if provided
        if kwargs.get("label") and len(args) >= 1:
            n_samples = len(args[0])
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.scatter(*args, **kwargs)

        # Track scatter data
        if len(args) >= 2:
            tracked_dict = {"scatter_df": pd.DataFrame({"x": args[0], "y": args[1]})}
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_errorbar(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib errorbar plot with tracking support."""
        method_name = "plot_errorbar"

        with self._no_tracking():
            result = self._axis_mpl.errorbar(*args, **kwargs)

        # Track errorbar data
        if len(args) >= 2:
            df_dict = {"x": args[0], "y": args[1]}
            if "yerr" in kwargs:
                df_dict["yerr"] = kwargs["yerr"]
            if "xerr" in kwargs:
                df_dict["xerr"] = kwargs["xerr"]
            tracked_dict = {"errorbar_df": pd.DataFrame(df_dict)}
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_fill_between(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib fill_between with tracking support."""
        method_name = "plot_fill_between"

        with self._no_tracking():
            result = self._axis_mpl.fill_between(*args, **kwargs)

        # Track fill_between data
        if len(args) >= 3:
            tracked_dict = {
                "fill_between_df": pd.DataFrame(
                    {"x": args[0], "y1": args[1], "y2": args[2] if len(args) > 2 else 0}
                )
            }
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_contour(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib contour plot with tracking support."""
        method_name = "plot_contour"

        with self._no_tracking():
            result = self._axis_mpl.contour(*args, **kwargs)

        # Track contour data
        if len(args) >= 3:
            # Flatten 2D arrays for CSV export
            X, Y, Z = args[0], args[1], args[2]
            tracked_dict = {
                "contour_df": pd.DataFrame(
                    {"X": np.ravel(X), "Y": np.ravel(Y), "Z": np.ravel(Z)}
                )
            }
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_imshow(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib imshow with tracking support."""
        method_name = "plot_imshow"

        with self._no_tracking():
            result = self._axis_mpl.imshow(*args, **kwargs)

        # Track image data
        if len(args) >= 1:
            # Create DataFrame with unique column names to avoid duplicates
            img_data = args[0]
            if hasattr(img_data, "shape") and len(img_data.shape) == 2:
                n_rows, n_cols = img_data.shape
                # Use column names like "col_0", "col_1", etc. instead of just integers
                df = pd.DataFrame(img_data, columns=[f"col_{i}" for i in range(n_cols)])
            else:
                df = pd.DataFrame(args[0])
            tracked_dict = {"imshow_df": df}
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_boxplot(
        self,
        *args,
        colors: Optional[List] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ):
        """Wrapper for matplotlib boxplot with tracking support and auto-styling."""
        method_name = "plot_boxplot"

        # Add sample size per group to label if provided (show range if variable)
        if kwargs.get("label") and len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                n_per_group = [len(g) for g in data]
                n_min, n_max = min(n_per_group), max(n_per_group)
                n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
                kwargs["label"] = f"{kwargs['label']} ($n$={n_str})"

        # Enable patch_artist for styling (fill colors, edges)
        if "patch_artist" not in kwargs:
            kwargs["patch_artist"] = True

        with self._no_tracking():
            result = self._axis_mpl.boxplot(*args, **kwargs)

        # Track boxplot data
        if len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                tracked_dict = {"boxplot_df": pd.DataFrame(data)}
            else:
                tracked_dict = {"boxplot_df": pd.DataFrame({"data": data})}
            self._track(track, id, method_name, tracked_dict, None)

        # Apply style_boxplot automatically for publication quality
        # Uses scitex palette by default, or custom colors if provided
        from scitex.plt.ax import style_boxplot

        style_boxplot(result, colors=colors)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_violinplot(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib violinplot with tracking support."""
        method_name = "plot_violinplot"

        # Add sample size per group to label if provided (show range if variable)
        if kwargs.get("label") and len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                n_per_group = [len(g) for g in data]
                n_min, n_max = min(n_per_group), max(n_per_group)
                n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
                kwargs["label"] = f"{kwargs['label']} ($n$={n_str})"

        with self._no_tracking():
            result = self._axis_mpl.violinplot(*args, **kwargs)

        # Track violin data
        if len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                tracked_dict = {"violinplot_df": pd.DataFrame(data)}
            else:
                tracked_dict = {"violinplot_df": pd.DataFrame({"data": data})}
            self._track(track, id, method_name, tracked_dict, None)

        # Apply post-processing (tick locator, spines, etc.)
        self._apply_scitex_postprocess(method_name, result, kwargs, args)

        return result



# EOF


# =============================================================================
# Deprecated plot_ aliases for stx_ methods (backward compatibility)
# These are defined outside the class to use the decorator properly
# =============================================================================
from scitex.decorators import deprecated


def _add_deprecated_aliases():
    """Add deprecated plot_ method aliases to MatplotlibPlotMixin."""
    deprecated_methods = [
        ("plot_image", "stx_image"),
        ("plot_kde", "stx_kde"),
        ("plot_conf_mat", "stx_conf_mat"),
        ("plot_rectangle", "stx_rectangle"),
        ("plot_fillv", "stx_fillv"),
        ("plot_box", "stx_box"),
        ("plot_raster", "stx_raster"),
        ("plot_ecdf", "stx_ecdf"),
        ("plot_joyplot", "stx_joyplot"),
        ("plot_line", "stx_line"),
        ("plot_scatter_hist", "stx_scatter_hist"),
        ("plot_heatmap", "stx_heatmap"),
        ("plot_violin", "stx_violin"),
        ("plot_mean_std", "stx_mean_std"),
        ("plot_mean_ci", "stx_mean_ci"),
        ("plot_median_iqr", "stx_median_iqr"),
        ("plot_shaded_line", "stx_shaded_line"),
    ]

    for old_name, new_name in deprecated_methods:
        def make_deprecated_method(target_name):
            @deprecated(reason=f"Use {target_name} instead")
            def method(self, *args, **kwargs):
                return getattr(self, target_name)(*args, **kwargs)
            return method

        setattr(MatplotlibPlotMixin, old_name, make_deprecated_method(new_name))


_add_deprecated_aliases()
