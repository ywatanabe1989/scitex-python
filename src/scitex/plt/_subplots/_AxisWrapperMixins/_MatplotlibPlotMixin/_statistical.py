#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _statistical.py - Statistical plot methods

"""Statistical plotting methods including line plots, box plots, and violin plots."""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scitex.types import ArrayLike

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


class StatisticalPlotMixin:
    """Mixin for statistical plotting methods."""

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
        method_name = "stx_rectangle"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_rectangle(
                self._axis_mpl, xx, yy, width, height, **kwargs
            )

        tracked_dict = {"x": xx, "y": yy, "width": width, "height": height}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

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
        method_name = "stx_fillv"

        with self._no_tracking():
            self._axis_mpl = self._get_ax_module().stx_fillv(
                self._axis_mpl, starts_1d, ends_1d, color=color, alpha=alpha
            )

        tracked_dict = {"starts": starts_1d, "ends": ends_1d}
        self._track(track, id, method_name, tracked_dict, None)
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
        method_name = "stx_box"

        _data = values_list.copy()

        if kwargs.get("label"):
            n_per_group = [len(g) for g in values_list]
            n_min, n_max = min(n_per_group), max(n_per_group)
            n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
            kwargs["label"] = kwargs["label"] + f" ($n$={n_str})"

        if "patch_artist" not in kwargs:
            kwargs["patch_artist"] = True

        with self._no_tracking():
            result = self._axis_mpl.boxplot(values_list, **kwargs)

        n_per_group = [len(g) for g in values_list]
        tracked_dict = {"data": _data, "n": n_per_group}
        self._track(track, id, method_name, tracked_dict, None)

        from scitex.plt.ax import style_boxplot
        style_boxplot(result, colors=colors)

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
        """Plot a histogram with bin alignment support."""
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
        method_name = "stx_violin"

        with self._no_tracking():
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
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl

    def stx_line(
        self,
        values_1d: ArrayLike,
        xx: Optional[ArrayLike] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot a simple line."""
        method_name = "stx_line"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_line(
                self._axis_mpl, values_1d, xx=xx, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, kwargs)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

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
        method_name = "stx_mean_std"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_mean_std(
                self._axis_mpl, values_2d, xx=xx, sd=sd, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

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
        method_name = "stx_mean_ci"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_mean_ci(
                self._axis_mpl, values_2d, xx=xx, perc=perc, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

    def stx_median_iqr(
        self,
        values_2d: ArrayLike,
        xx: Optional[ArrayLike] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot median line with interquartile range shading."""
        method_name = "stx_median_iqr"

        with self._no_tracking():
            self._axis_mpl, plot_df = self._get_ax_module().stx_median_iqr(
                self._axis_mpl, values_2d, xx=xx, **kwargs
            )

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df

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
        method_name = "stx_shaded_line"

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

        tracked_dict = {"plot_df": plot_df}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name)

        return self._axis_mpl, plot_df


# EOF
