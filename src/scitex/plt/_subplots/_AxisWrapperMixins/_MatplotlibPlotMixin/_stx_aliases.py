#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _stx_aliases.py - stx_ aliases for standard matplotlib methods

"""stx_ prefixed aliases for standard matplotlib methods with tracking support."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from scitex.types import ArrayLike

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


class StxAliasesMixin:
    """Mixin providing stx_ aliases for standard matplotlib methods."""

    def stx_bar(
        self, x, height, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Bar plot with scitex styling and tracking."""
        method_name = "stx_bar"

        if kwargs.get("label"):
            n_samples = len(x)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.bar(x, height, **kwargs)

        tracked_dict = {"bar_df": pd.DataFrame({"x": x, "height": height})}
        self._track(track, id, method_name, tracked_dict, None)

        from scitex.plt.ax import style_barplot
        style_barplot(result)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_barh(
        self, y, width, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Horizontal bar plot with scitex styling and tracking."""
        method_name = "stx_barh"

        if kwargs.get("label"):
            n_samples = len(y)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.barh(y, width, **kwargs)

        tracked_dict = {"barh_df": pd.DataFrame({"y": y, "width": width})}
        self._track(track, id, method_name, tracked_dict, None)
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_scatter(self, x, y, track: bool = True, id: Optional[str] = None, **kwargs):
        """Scatter plot with scitex styling and tracking."""
        method_name = "stx_scatter"

        if kwargs.get("label"):
            n_samples = len(x)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.scatter(x, y, **kwargs)

        tracked_dict = {"scatter_df": pd.DataFrame({"x": x, "y": y})}
        self._track(track, id, method_name, tracked_dict, None)

        from scitex.plt.ax import style_scatter
        style_scatter(result)

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
        """Error bar plot with scitex styling and tracking."""
        method_name = "stx_errorbar"

        if kwargs.get("label"):
            n_samples = len(x)
            kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"

        with self._no_tracking():
            result = self._axis_mpl.errorbar(x, y, yerr=yerr, xerr=xerr, **kwargs)

        df_dict = {"x": x, "y": y}
        if yerr is not None:
            df_dict["yerr"] = yerr
        if xerr is not None:
            df_dict["xerr"] = xerr
        tracked_dict = {"errorbar_df": pd.DataFrame(df_dict)}
        self._track(track, id, method_name, tracked_dict, None)

        from scitex.plt.ax import style_errorbar
        style_errorbar(result)

        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_fill_between(
        self, x, y1, y2=0, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Fill between plot with scitex styling and tracking."""
        method_name = "stx_fill_between"

        with self._no_tracking():
            result = self._axis_mpl.fill_between(x, y1, y2, **kwargs)

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
        self._apply_scitex_postprocess(method_name, result)

        return result

    def stx_contour(
        self, *args, track: bool = True, id: Optional[str] = None, **kwargs
    ):
        """Contour plot with scitex styling and tracking."""
        method_name = "stx_contour"

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

    def stx_imshow(self, data, track: bool = True, id: Optional[str] = None, **kwargs):
        """Image display with scitex styling and tracking."""
        method_name = "stx_imshow"

        with self._no_tracking():
            result = self._axis_mpl.imshow(data, **kwargs)

        if hasattr(data, "shape") and len(data.shape) == 2:
            n_rows, n_cols = data.shape
            df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])
        else:
            df = pd.DataFrame(data)
        tracked_dict = {"imshow_df": df}
        self._track(track, id, method_name, tracked_dict, None)

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
        """Boxplot with scitex styling and tracking (alias for stx_box)."""
        return self.stx_box(data, colors=colors, track=track, id=id, **kwargs)

    def stx_violinplot(
        self,
        data,
        colors: Optional[List] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ):
        """Violinplot with scitex styling and tracking (alias for stx_violin)."""
        return self.stx_violin(data, colors=colors, track=track, id=id, **kwargs)


# EOF
