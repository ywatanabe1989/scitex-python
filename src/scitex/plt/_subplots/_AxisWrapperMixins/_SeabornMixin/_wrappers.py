#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: _wrappers.py - Seaborn plot wrappers

"""Seaborn plot wrappers with scitex integration."""

import os

import scitex
import numpy as np
import seaborn as sns

from ._base import sns_copy_doc

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


class SeabornWrappersMixin:
    """Mixin providing sns_ prefixed seaborn wrappers."""

    def _get_ax_module(self):
        """Lazy import ax module to avoid circular imports."""
        from .....plt import ax as ax_module
        return ax_module

    @sns_copy_doc
    def sns_barplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_barplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_boxplot(
        self,
        data=None,
        x=None,
        y=None,
        strip=False,
        track=True,
        id=None,
        **kwargs,
    ):
        self._sns_base_xyhue(
            "sns_boxplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

        # Post-processing: Style boxplot with black medians
        from scitex.plt.utils import mm_to_pt
        lw_pt = mm_to_pt(0.2)

        for line in self._axis_mpl.get_lines():
            line.set_linewidth(lw_pt)
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) == 2 and len(ydata) == 2:
                if ydata[0] == ydata[1]:
                    x_span = abs(xdata[1] - xdata[0])
                    if x_span < 0.4:
                        line.set_color("black")

        if strip:
            strip_kwargs = kwargs.copy()
            strip_kwargs.pop("notch", None)
            strip_kwargs.pop("whis", None)
            self.sns_stripplot(
                data=data,
                x=x,
                y=y,
                track=False,
                id=f"{id}_strip",
                **strip_kwargs,
            )

    @sns_copy_doc
    def sns_heatmap(self, *args, xyz=False, track=True, id=None, **kwargs):
        method_name = "sns_heatmap"
        df = args[0]
        if xyz:
            df = scitex.pd.to_xyz(df)
        self._sns_base(method_name, *args, track=track, track_obj=df, id=id, **kwargs)

    @sns_copy_doc
    def sns_histplot(
        self,
        data=None,
        x=None,
        y=None,
        bins=10,
        align_bins=True,
        track=True,
        id=None,
        **kwargs,
    ):
        """Plot histogram with bin alignment support."""
        method_name = "sns_histplot"

        plot_data = None
        if data is not None and x is not None:
            plot_data = (
                data[x].values
                if hasattr(data, "columns") and x in data.columns
                else None
            )

        axis_id = str(hash(self._axis_mpl))
        hist_id = id if id is not None else str(self.id)
        range_value = kwargs.get("binrange", None)

        if align_bins and plot_data is not None:
            from .....plt.utils import histogram_bin_manager
            bins_val, range_val = histogram_bin_manager.register_histogram(
                axis_id, hist_id, plot_data, bins, range_value
            )
            kwargs["bins"] = bins_val
            if range_value is not None:
                kwargs["binrange"] = range_val

        with self._no_tracking():
            sns_plot = sns.histplot(data=data, x=x, y=y, ax=self._axis_mpl, **kwargs)

            hist_result = None
            if hasattr(sns_plot, "patches") and sns_plot.patches:
                patches = sns_plot.patches
                if patches:
                    counts = np.array([p.get_height() for p in patches])
                    bin_edges = []
                    for p in patches:
                        bin_edges.append(p.get_x())
                    if patches:
                        bin_edges.append(patches[-1].get_x() + patches[-1].get_width())
                    hist_result = (counts, np.array(bin_edges))

        track_obj = self._sns_prepare_xyhue(data, x, y, kwargs.get("hue"))
        tracked_dict = {
            "data": track_obj,
            "args": (data, x, y),
            "hist_result": hist_result,
        }
        self._track(track, id, method_name, tracked_dict, kwargs)

        return sns_plot

    @sns_copy_doc
    def sns_kdeplot(
        self,
        data=None,
        x=None,
        y=None,
        xlim=None,
        ylim=None,
        track=True,
        id=None,
        **kwargs,
    ):
        hue_col = kwargs.pop("hue", None)

        if hue_col:
            hues = data[hue_col]
            if x is not None:
                lim = xlim
                for hue in np.unique(hues):
                    _data = data.loc[hues == hue, x]
                    self.stx_kde(_data, xlim=lim, label=hue, id=hue, **kwargs)
            if y is not None:
                lim = ylim
                for hue in np.unique(hues):
                    _data = data.loc[hues == hue, y]
                    self.stx_kde(_data, xlim=lim, label=hue, id=hue, **kwargs)
        else:
            if x is not None:
                _data, lim = data[x], xlim
            if y is not None:
                _data, lim = data[y], ylim
            self.stx_kde(_data, xlim=lim, **kwargs)

    @sns_copy_doc
    def sns_pairplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_pairplot", *args, track=track, id=id, **kwargs)

    @sns_copy_doc
    def sns_scatterplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_scatterplot",
            data=data,
            x=x,
            y=y,
            track=track,
            id=id,
            **kwargs,
        )

    @sns_copy_doc
    def sns_lineplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_lineplot",
            data=data,
            x=x,
            y=y,
            track=track,
            id=id,
            **kwargs,
        )

    @sns_copy_doc
    def sns_swarmplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_swarmplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_stripplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_stripplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_violinplot(
        self,
        data=None,
        x=None,
        y=None,
        track=True,
        id=None,
        half=False,
        **kwargs,
    ):
        if half:
            with self._no_tracking():
                self._axis_mpl = self._get_ax_module().plot_half_violin(
                    self._axis_mpl, data=data, x=x, y=y, **kwargs
                )
        else:
            self._sns_base_xyhue(
                "sns_violinplot",
                data=data,
                x=x,
                y=y,
                track=track,
                id=id,
                **kwargs,
            )

        track_obj = self._sns_prepare_xyhue(data, x, y, kwargs.get("hue"))
        self._track(track, id, "sns_violinplot", track_obj, kwargs)

        return self._axis_mpl

    @sns_copy_doc
    def sns_jointplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_jointplot", *args, track=track, id=id, **kwargs)


# EOF
