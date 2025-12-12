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
    """Mixin for scientific/specialized plotting methods."""

    def stx_image(
        self,
        arr_2d: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "stx_image"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_image(
                self._axis_mpl, arr_2d, **kwargs
            )

        tracked_dict = {"image_df": pd.DataFrame(arr_2d)}
        if kwargs.get("xyz", False):
            tracked_dict["image_df"] = to_xyz(tracked_dict["image_df"])
        self._track(track, id, method_name, tracked_dict, None)
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
        method_name = "stx_kde"

        n_samples = (~np.isnan(values_1d)).sum()
        if kwargs.get("label"):
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        xlim = kwargs.pop("xlim", None)
        if not xlim:
            xlim = (np.nanmin(values_1d), np.nanmax(values_1d))

        xx = np.linspace(xlim[0], xlim[1], int(1e3))
        density = gaussian_kde(values_1d)(xx)
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
        method_name = "stx_conf_mat"

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
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, bacc_val

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
        method_name = "stx_raster"

        with self._no_tracking():
            self._axis_mpl, raster_digit_df = self._get_ax_module().stx_raster(
                self._axis_mpl, spike_times_list, time=time
            )

        tracked_dict = {"raster_digit_df": raster_digit_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, raster_digit_df

    def stx_ecdf(
        self,
        values_1d: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "stx_ecdf"

        with self._no_tracking():
            self._axis_mpl, ecdf_df = self._get_ax_module().stx_ecdf(
                self._axis_mpl, values_1d, **kwargs
            )

        tracked_dict = {"ecdf_df": ecdf_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, ecdf_df

    def stx_joyplot(
        self,
        arrays: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "stx_joyplot"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_joyplot(
                self._axis_mpl, arrays, **kwargs
            )

        tracked_dict = {"joyplot_data": arrays}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

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
        method_name = "stx_heatmap"

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

        tracked_dict = {
            "data": values_2d,
            "x_labels": x_labels,
            "y_labels": y_labels,
        }
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return ax, im, cbar


# EOF
