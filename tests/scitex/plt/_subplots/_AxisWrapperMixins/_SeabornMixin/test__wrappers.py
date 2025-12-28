# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_SeabornMixin/_wrappers.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: _wrappers.py - Seaborn plot wrappers
# 
# """Seaborn plot wrappers with SciTeX integration."""
# 
# import os
# from typing import Optional, Union
# 
# import numpy as np
# import pandas as pd
# import seaborn as sns
# 
# from scitex.types import ArrayLike
# 
# from ._base import sns_copy_doc
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# 
# class SeabornWrappersMixin:
#     """Mixin providing sns_ prefixed seaborn wrappers.
# 
#     All methods use the seaborn DataFrame-centric interface:
#     - data: DataFrame containing the data
#     - x, y: Column names for axes
#     - hue: Column name for color grouping
# 
#     These methods integrate with SciTeX tracking and styling.
#     """
# 
#     def _get_ax_module(self):
#         """Lazy import ax module to avoid circular imports."""
#         from .....plt import ax as ax_module
# 
#         return ax_module
# 
#     @sns_copy_doc
#     def sns_barplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a bar plot showing point estimates and error bars.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis categories.
#         y : str, optional
#             Column name for y-axis values.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.barplot`.
# 
#         See Also
#         --------
#         stx_bar : Array-based bar plot.
#         """
#         self._sns_base_xyhue(
#             "sns_barplot", data=data, x=x, y=y, track=track, id=id, **kwargs
#         )
# 
#     @sns_copy_doc
#     def sns_boxplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         strip: bool = False,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a box plot showing distributions with quartiles.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis grouping.
#         y : str, optional
#             Column name for y-axis values.
#         strip : bool, default False
#             If True, overlay a stripplot showing individual points.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.boxplot`.
# 
#         See Also
#         --------
#         stx_box : Array-based boxplot.
#         sns_violinplot : Violin plot alternative.
#         """
#         self._sns_base_xyhue(
#             "sns_boxplot", data=data, x=x, y=y, track=track, id=id, **kwargs
#         )
# 
#         # Post-processing: Style boxplot with black medians
#         from scitex.plt.utils import mm_to_pt
# 
#         lw_pt = mm_to_pt(0.2)
# 
#         for line in self._axis_mpl.get_lines():
#             line.set_linewidth(lw_pt)
#             xdata = line.get_xdata()
#             ydata = line.get_ydata()
#             if len(xdata) == 2 and len(ydata) == 2:
#                 if ydata[0] == ydata[1]:
#                     x_span = abs(xdata[1] - xdata[0])
#                     if x_span < 0.4:
#                         line.set_color("black")
# 
#         if strip:
#             strip_kwargs = kwargs.copy()
#             strip_kwargs.pop("notch", None)
#             strip_kwargs.pop("whis", None)
#             self.sns_stripplot(
#                 data=data,
#                 x=x,
#                 y=y,
#                 track=False,
#                 id=f"{id}_strip",
#                 **strip_kwargs,
#             )
# 
#     @sns_copy_doc
#     def sns_heatmap(
#         self,
#         data: Union[pd.DataFrame, ArrayLike],
#         *,
#         xyz: bool = False,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a heatmap from rectangular data.
# 
#         Parameters
#         ----------
#         data : DataFrame or array-like
#             2D dataset for the heatmap.
#         xyz : bool, default False
#             If True, convert data to XYZ format before plotting.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.heatmap`.
# 
#         See Also
#         --------
#         stx_heatmap : Array-based annotated heatmap.
#         stx_image : Simple image display.
#         """
#         import scitex
# 
#         method_name = "sns_heatmap"
#         df = data
#         if xyz:
#             df = scitex.pd.to_xyz(df)
#         self._sns_base(method_name, df, track=track, track_obj=df, id=id, **kwargs)
# 
#     @sns_copy_doc
#     def sns_histplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         bins: int = 10,
#         align_bins: bool = True,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a histogram with optional kernel density estimate.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis values.
#         y : str, optional
#             Column name for y-axis values.
#         bins : int, default 10
#             Number of histogram bins.
#         align_bins : bool, default True
#             Align bins across multiple histograms on same axes.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.histplot`.
#             Common options: kde, stat, element, hue.
# 
#         See Also
#         --------
#         hist : Array-based histogram.
#         stx_kde : Kernel density estimate.
#         """
#         method_name = "sns_histplot"
# 
#         plot_data = None
#         if data is not None and x is not None:
#             plot_data = (
#                 data[x].values
#                 if hasattr(data, "columns") and x in data.columns
#                 else None
#             )
# 
#         axis_id = str(hash(self._axis_mpl))
#         hist_id = id if id is not None else str(self.id)
#         range_value = kwargs.get("binrange", None)
# 
#         if align_bins and plot_data is not None:
#             from .....plt.utils import histogram_bin_manager
# 
#             bins_val, range_val = histogram_bin_manager.register_histogram(
#                 axis_id, hist_id, plot_data, bins, range_value
#             )
#             kwargs["bins"] = bins_val
#             if range_value is not None:
#                 kwargs["binrange"] = range_val
# 
#         with self._no_tracking():
#             sns_plot = sns.histplot(data=data, x=x, y=y, ax=self._axis_mpl, **kwargs)
# 
#             hist_result = None
#             if hasattr(sns_plot, "patches") and sns_plot.patches:
#                 patches = sns_plot.patches
#                 if patches:
#                     counts = np.array([p.get_height() for p in patches])
#                     bin_edges = []
#                     for p in patches:
#                         bin_edges.append(p.get_x())
#                     if patches:
#                         bin_edges.append(patches[-1].get_x() + patches[-1].get_width())
#                     hist_result = (counts, np.array(bin_edges))
# 
#         track_obj = self._sns_prepare_xyhue(data, x, y, kwargs.get("hue"))
#         tracked_dict = {
#             "data": track_obj,
#             "args": (data, x, y),
#             "hist_result": hist_result,
#         }
#         self._track(track, id, method_name, tracked_dict, kwargs)
# 
#         return sns_plot
# 
#     @sns_copy_doc
#     def sns_kdeplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         xlim: Optional[tuple] = None,
#         ylim: Optional[tuple] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a kernel density estimate plot.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis values.
#         y : str, optional
#             Column name for y-axis values.
#         xlim : tuple, optional
#             Limits for x-axis KDE range.
#         ylim : tuple, optional
#             Limits for y-axis KDE range.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to the KDE function.
# 
#         See Also
#         --------
#         stx_kde : Array-based KDE plot.
#         sns_histplot : Histogram with optional KDE.
#         """
#         hue_col = kwargs.pop("hue", None)
# 
#         if hue_col:
#             hues = data[hue_col]
#             if x is not None:
#                 lim = xlim
#                 for hue in np.unique(hues):
#                     _data = data.loc[hues == hue, x]
#                     self.stx_kde(_data, xlim=lim, label=hue, id=hue, **kwargs)
#             if y is not None:
#                 lim = ylim
#                 for hue in np.unique(hues):
#                     _data = data.loc[hues == hue, y]
#                     self.stx_kde(_data, xlim=lim, label=hue, id=hue, **kwargs)
#         else:
#             if x is not None:
#                 _data, lim = data[x], xlim
#             if y is not None:
#                 _data, lim = data[y], ylim
#             self.stx_kde(_data, xlim=lim, **kwargs)
# 
#     @sns_copy_doc
#     def sns_pairplot(
#         self,
#         *args,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a grid of pairwise relationships in a dataset.
# 
#         Parameters
#         ----------
#         *args
#             Positional arguments passed to `seaborn.pairplot`.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.pairplot`.
#         """
#         self._sns_base("sns_pairplot", *args, track=track, id=id, **kwargs)
# 
#     @sns_copy_doc
#     def sns_scatterplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a scatter plot with semantic mappings.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis values.
#         y : str, optional
#             Column name for y-axis values.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.scatterplot`.
#             Common options: hue, size, style.
# 
#         See Also
#         --------
#         stx_scatter : Array-based scatter plot.
#         """
#         self._sns_base_xyhue(
#             "sns_scatterplot",
#             data=data,
#             x=x,
#             y=y,
#             track=track,
#             id=id,
#             **kwargs,
#         )
# 
#     @sns_copy_doc
#     def sns_lineplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a line plot with semantic mappings.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis values.
#         y : str, optional
#             Column name for y-axis values.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.lineplot`.
#             Common options: hue, size, style, estimator.
# 
#         See Also
#         --------
#         stx_line : Array-based line plot.
#         stx_mean_std : Line with uncertainty shading.
#         """
#         self._sns_base_xyhue(
#             "sns_lineplot",
#             data=data,
#             x=x,
#             y=y,
#             track=track,
#             id=id,
#             **kwargs,
#         )
# 
#     @sns_copy_doc
#     def sns_swarmplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a categorical scatter plot with non-overlapping points.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis grouping.
#         y : str, optional
#             Column name for y-axis values.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.swarmplot`.
# 
#         See Also
#         --------
#         sns_stripplot : Jittered categorical scatter.
#         sns_boxplot : Box plot for distributions.
#         """
#         self._sns_base_xyhue(
#             "sns_swarmplot", data=data, x=x, y=y, track=track, id=id, **kwargs
#         )
# 
#     @sns_copy_doc
#     def sns_stripplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a categorical scatter plot with jittered points.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis grouping.
#         y : str, optional
#             Column name for y-axis values.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.stripplot`.
# 
#         See Also
#         --------
#         sns_swarmplot : Non-overlapping categorical scatter.
#         sns_boxplot : Often combined with stripplot.
#         """
#         self._sns_base_xyhue(
#             "sns_stripplot", data=data, x=x, y=y, track=track, id=id, **kwargs
#         )
# 
#     @sns_copy_doc
#     def sns_violinplot(
#         self,
#         data: Optional[pd.DataFrame] = None,
#         *,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         half: bool = False,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a violin plot combining box plot with kernel density.
# 
#         Parameters
#         ----------
#         data : DataFrame, optional
#             Input data structure.
#         x : str, optional
#             Column name for x-axis grouping.
#         y : str, optional
#             Column name for y-axis values.
#         half : bool, default False
#             If True, draw half-violins (useful for paired comparisons).
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.violinplot`.
# 
#         See Also
#         --------
#         stx_violin : Array-based violin plot.
#         sns_boxplot : Box plot alternative.
#         """
#         if half:
#             with self._no_tracking():
#                 self._axis_mpl = self._get_ax_module().plot_half_violin(
#                     self._axis_mpl, data=data, x=x, y=y, **kwargs
#                 )
#         else:
#             self._sns_base_xyhue(
#                 "sns_violinplot",
#                 data=data,
#                 x=x,
#                 y=y,
#                 track=track,
#                 id=id,
#                 **kwargs,
#             )
# 
#         track_obj = self._sns_prepare_xyhue(data, x, y, kwargs.get("hue"))
#         self._track(track, id, "sns_violinplot", track_obj, kwargs)
# 
#         return self._axis_mpl
# 
#     @sns_copy_doc
#     def sns_jointplot(
#         self,
#         *args,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create a figure with joint and marginal distributions.
# 
#         Parameters
#         ----------
#         *args
#             Positional arguments passed to `seaborn.jointplot`.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `seaborn.jointplot`.
# 
#         See Also
#         --------
#         stx_scatter_hist : Array-based scatter with marginal histograms.
#         """
#         self._sns_base("sns_jointplot", *args, track=track, id=id, **kwargs)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_SeabornMixin/_wrappers.py
# --------------------------------------------------------------------------------
