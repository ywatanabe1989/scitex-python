#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _plot_wrappers.py - Standard matplotlib plot_ wrappers

"""Standard matplotlib wrappers with plot_ prefix and tracking support."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


class PlotWrappersMixin:
    """Mixin providing plot_ prefixed matplotlib wrappers."""

    def plot_bar(self, *args, track: bool = True, id: Optional[str] = None, **kwargs):
        """Wrapper for matplotlib bar plot with tracking support."""
        method_name = "plot_bar"

        with self._no_tracking():
            result = self._axis_mpl.bar(*args, **kwargs)

        if len(args) >= 2:
            tracked_dict = {"bar_df": pd.DataFrame({"x": args[0], "height": args[1]})}
            self._track(track, id, method_name, tracked_dict, None)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_barh(self, *args, track: bool = True, id: Optional[str] = None, **kwargs):
        """Wrapper for matplotlib horizontal bar plot with tracking support."""
        method_name = "plot_barh"

        with self._no_tracking():
            result = self._axis_mpl.barh(*args, **kwargs)

        if len(args) >= 2:
            tracked_dict = {"barh_df": pd.DataFrame({"y": args[0], "width": args[1]})}
            self._track(track, id, method_name, tracked_dict, None)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_scatter(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib scatter plot with tracking support."""
        method_name = "plot_scatter"

        if kwargs.get("label") and len(args) >= 1:
            n_samples = len(args[0])
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.scatter(*args, **kwargs)

        if len(args) >= 2:
            tracked_dict = {"scatter_df": pd.DataFrame({"x": args[0], "y": args[1]})}
            self._track(track, id, method_name, tracked_dict, None)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_errorbar(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib errorbar plot with tracking support."""
        method_name = "plot_errorbar"

        with self._no_tracking():
            result = self._axis_mpl.errorbar(*args, **kwargs)

        if len(args) >= 2:
            df_dict = {"x": args[0], "y": args[1]}
            if "yerr" in kwargs:
                df_dict["yerr"] = kwargs["yerr"]
            if "xerr" in kwargs:
                df_dict["xerr"] = kwargs["xerr"]
            tracked_dict = {"errorbar_df": pd.DataFrame(df_dict)}
            self._track(track, id, method_name, tracked_dict, None)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_fill_between(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib fill_between with tracking support."""
        method_name = "plot_fill_between"

        with self._no_tracking():
            result = self._axis_mpl.fill_between(*args, **kwargs)

        if len(args) >= 3:
            tracked_dict = {
                "fill_between_df": pd.DataFrame(
                    {"x": args[0], "y1": args[1], "y2": args[2] if len(args) > 2 else 0}
                )
            }
            self._track(track, id, method_name, tracked_dict, None)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_contour(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib contour plot with tracking support."""
        method_name = "plot_contour"

        with self._no_tracking():
            result = self._axis_mpl.contour(*args, **kwargs)

        if len(args) >= 3:
            X, Y, Z = args[0], args[1], args[2]
            tracked_dict = {
                "contour_df": pd.DataFrame(
                    {"X": np.ravel(X), "Y": np.ravel(Y), "Z": np.ravel(Z)}
                )
            }
            self._track(track, id, method_name, tracked_dict, None)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_imshow(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib imshow with tracking support."""
        method_name = "plot_imshow"

        with self._no_tracking():
            result = self._axis_mpl.imshow(*args, **kwargs)

        if len(args) >= 1:
            img_data = args[0]
            if hasattr(img_data, "shape") and len(img_data.shape) == 2:
                n_rows, n_cols = img_data.shape
                df = pd.DataFrame(img_data, columns=[f"col_{i}" for i in range(n_cols)])
            else:
                df = pd.DataFrame(args[0])
            tracked_dict = {"imshow_df": df}
            self._track(track, id, method_name, tracked_dict, None)

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

        if kwargs.get("label") and len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                n_per_group = [len(g) for g in data]
                n_min, n_max = min(n_per_group), max(n_per_group)
                n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
                kwargs["label"] = f"{kwargs['label']} ($n$={n_str})"

        if "patch_artist" not in kwargs:
            kwargs["patch_artist"] = True

        with self._no_tracking():
            result = self._axis_mpl.boxplot(*args, **kwargs)

        if len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                tracked_dict = {"boxplot_df": pd.DataFrame(data)}
            else:
                tracked_dict = {"boxplot_df": pd.DataFrame({"data": data})}
            self._track(track, id, method_name, tracked_dict, None)

        from scitex.plt.ax import style_boxplot
        style_boxplot(result, colors=colors)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def plot_violinplot(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Wrapper for matplotlib violinplot with tracking support."""
        method_name = "plot_violinplot"

        if kwargs.get("label") and len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                n_per_group = [len(g) for g in data]
                n_min, n_max = min(n_per_group), max(n_per_group)
                n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
                kwargs["label"] = f"{kwargs['label']} ($n$={n_str})"

        with self._no_tracking():
            result = self._axis_mpl.violinplot(*args, **kwargs)

        if len(args) >= 1:
            data = args[0]
            if isinstance(data, list):
                tracked_dict = {"violinplot_df": pd.DataFrame(data)}
            else:
                tracked_dict = {"violinplot_df": pd.DataFrame({"data": data})}
            self._track(track, id, method_name, tracked_dict, None)

        self._apply_scitex_postprocess(method_name, result, kwargs, args)

        return result


# EOF
