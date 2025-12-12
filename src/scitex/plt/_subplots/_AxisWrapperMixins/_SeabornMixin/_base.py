#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _base.py - Base seaborn functionality

"""Base seaborn mixin with helper methods for tracking and data preparation."""

import os
from functools import wraps

import scitex
import numpy as np
import pandas as pd
import seaborn as sns

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


def sns_copy_doc(func):
    """Decorator to copy docstring from seaborn function."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    sns_method_name = func.__name__.split("sns_")[-1]
    wrapper.__doc__ = getattr(sns, sns_method_name).__doc__
    return wrapper


class SeabornBaseMixin:
    """Base mixin for seaborn integration with tracking support."""

    def _sns_base(
        self, method_name, *args, track=True, track_obj=None, id=None, **kwargs
    ):
        """Execute seaborn plot method with tracking support."""
        sns_method_name = method_name.split("sns_")[-1]

        with self._no_tracking():
            sns_plot_fn = getattr(sns, sns_method_name)

            if kwargs.get("hue_colors"):
                kwargs = scitex.gen.alternate_kwarg(
                    kwargs, primary_key="palette", alternate_key="hue_colors"
                )

            import warnings
            from scitex import logging

            mpl_logger = logging.getLogger("matplotlib")
            original_level = mpl_logger.level
            mpl_logger.setLevel(logging.WARNING)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*categorical units.*parsable as floats or dates.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Using categorical units.*",
                        module="matplotlib.*",
                    )
                    warnings.simplefilter("ignore", UserWarning)

                    self._axis_mpl = sns_plot_fn(ax=self._axis_mpl, *args, **kwargs)
            finally:
                mpl_logger.setLevel(original_level)

            # Post-processing for histplot with kde=True
            if sns_method_name == "histplot" and kwargs.get("kde", False):
                from scitex.plt.utils import mm_to_pt
                kde_lw = mm_to_pt(0.2)
                for line in self._axis_mpl.get_lines():
                    line.set_linewidth(kde_lw)
                    line.set_color("black")
                    line.set_linestyle("--")

            # Post-processing for histplot alpha
            if sns_method_name == "histplot" and "alpha" not in kwargs:
                for patch in self._axis_mpl.patches:
                    patch.set_alpha(1.0)

        track_obj = track_obj if track_obj is not None else args
        tracked_dict = {
            "data": track_obj,
            "args": args,
        }
        self._track(track, id, method_name, tracked_dict, kwargs)

    def _sns_base_xyhue(self, method_name, *args, track=True, id=None, **kwargs):
        """Execute seaborn plot with x/y/hue data preparation."""
        df = kwargs.get("data")
        x, y, hue = kwargs.get("x"), kwargs.get("y"), kwargs.get("hue")

        track_obj = self._sns_prepare_xyhue(df, x, y, hue) if df is not None else None
        self._sns_base(
            method_name,
            *args,
            track=track,
            track_obj=track_obj,
            id=id,
            **kwargs,
        )

    def _sns_prepare_xyhue(self, data=None, x=None, y=None, hue=None, **kwargs):
        """Prepare data for tracking based on x/y/hue configuration."""
        data = data.reset_index()

        if hue is not None:
            if x is None and y is None:
                return data
            elif x is None:
                agg_dict = {}
                for hh in data[hue].unique():
                    agg_dict[hh] = data.loc[data[hue] == hh, y]
                df = scitex.pd.force_df(agg_dict)
                return df
            elif y is None:
                df = pd.concat(
                    [data.loc[data[hue] == hh, x] for hh in data[hue].unique()],
                    axis=1,
                )
                return df
            else:
                pivoted_data = data.pivot_table(
                    values=y,
                    index=data.index,
                    columns=[x, hue],
                    aggfunc="first",
                )
                pivoted_data.columns = [
                    f"{col[0]}-{col[1]}" for col in pivoted_data.columns
                ]
                return pivoted_data
        else:
            if x is None and y is None:
                return data
            elif x is None:
                return data[[y]]
            elif y is None:
                return data[[x]]
            else:
                return data.pivot_table(
                    values=y, index=data.index, columns=x, aggfunc="first"
                )


# EOF
