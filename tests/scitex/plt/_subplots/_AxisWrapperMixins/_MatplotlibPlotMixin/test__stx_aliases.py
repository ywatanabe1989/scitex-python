# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_MatplotlibPlotMixin/_stx_aliases.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: _stx_aliases.py - stx_ aliases for standard matplotlib methods
# 
# """stx_ prefixed aliases for standard matplotlib methods with tracking support."""
# 
# import os
# from typing import List, Optional, Sequence, Union
# 
# import numpy as np
# import pandas as pd
# from matplotlib.container import BarContainer
# from matplotlib.collections import PathCollection
# from matplotlib.contour import QuadContourSet
# from matplotlib.image import AxesImage
# 
# from scitex.types import ArrayLike
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# 
# class StxAliasesMixin:
#     """Mixin providing stx_ aliases for standard matplotlib methods.
# 
#     These methods wrap standard matplotlib plotting functions with:
#     - SciTeX styling applied automatically
#     - Data tracking for reproducibility
#     - Sample size annotations in labels
#     """
# 
#     def stx_bar(
#         self,
#         x: ArrayLike,
#         height: ArrayLike,
#         *,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> BarContainer:
#         """Create a bar plot with SciTeX styling and tracking.
# 
#         Parameters
#         ----------
#         x : array-like
#             X coordinates of the bars.
#         height : array-like
#             Heights of the bars.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.bar`.
# 
#         Returns
#         -------
#         BarContainer
#             Container with all the bars.
# 
#         See Also
#         --------
#         stx_barh : Horizontal bar plot.
#         mpl_bar : Raw matplotlib bar without styling.
# 
#         Examples
#         --------
#         >>> ax.stx_bar([1, 2, 3], [4, 5, 6])
#         >>> ax.stx_bar(x, height, label="Group A", color="blue")
#         """
#         method_name = "stx_bar"
# 
#         if kwargs.get("label"):
#             n_samples = len(x)
#             kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"
# 
#         with self._no_tracking():
#             result = self._axis_mpl.bar(x, height, **kwargs)
# 
#         tracked_dict = {"bar_df": pd.DataFrame({"x": x, "height": height})}
#         self._track(track, id, method_name, tracked_dict, None)
# 
#         from scitex.plt.ax import style_barplot
# 
#         style_barplot(result)
# 
#         self._apply_scitex_postprocess(method_name, result)
# 
#         return result
# 
#     def stx_barh(
#         self,
#         y: ArrayLike,
#         width: ArrayLike,
#         *,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> BarContainer:
#         """Create a horizontal bar plot with SciTeX styling and tracking.
# 
#         Parameters
#         ----------
#         y : array-like
#             Y coordinates of the bars.
#         width : array-like
#             Widths of the bars.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.barh`.
# 
#         Returns
#         -------
#         BarContainer
#             Container with all the bars.
# 
#         See Also
#         --------
#         stx_bar : Vertical bar plot.
#         mpl_barh : Raw matplotlib barh without styling.
# 
#         Examples
#         --------
#         >>> ax.stx_barh([1, 2, 3], [4, 5, 6])
#         """
#         method_name = "stx_barh"
# 
#         if kwargs.get("label"):
#             n_samples = len(y)
#             kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"
# 
#         with self._no_tracking():
#             result = self._axis_mpl.barh(y, width, **kwargs)
# 
#         tracked_dict = {"barh_df": pd.DataFrame({"y": y, "width": width})}
#         self._track(track, id, method_name, tracked_dict, None)
#         self._apply_scitex_postprocess(method_name, result)
# 
#         return result
# 
#     def stx_scatter(
#         self,
#         x: ArrayLike,
#         y: ArrayLike,
#         *,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> PathCollection:
#         """Create a scatter plot with SciTeX styling and tracking.
# 
#         Parameters
#         ----------
#         x : array-like
#             X coordinates of the data points.
#         y : array-like
#             Y coordinates of the data points.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.scatter`.
# 
#         Returns
#         -------
#         PathCollection
#             Collection of scatter points.
# 
#         See Also
#         --------
#         sns_scatterplot : DataFrame-based scatter plot.
#         mpl_scatter : Raw matplotlib scatter without styling.
# 
#         Examples
#         --------
#         >>> ax.stx_scatter(x, y, label="Data", s=50)
#         """
#         method_name = "stx_scatter"
# 
#         if kwargs.get("label"):
#             n_samples = len(x)
#             kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"
# 
#         with self._no_tracking():
#             result = self._axis_mpl.scatter(x, y, **kwargs)
# 
#         tracked_dict = {"scatter_df": pd.DataFrame({"x": x, "y": y})}
#         self._track(track, id, method_name, tracked_dict, None)
# 
#         from scitex.plt.ax import style_scatter
# 
#         style_scatter(result)
# 
#         self._apply_scitex_postprocess(method_name, result)
# 
#         return result
# 
#     def stx_errorbar(
#         self,
#         x: ArrayLike,
#         y: ArrayLike,
#         *,
#         yerr: Optional[ArrayLike] = None,
#         xerr: Optional[ArrayLike] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Create an error bar plot with SciTeX styling and tracking.
# 
#         Parameters
#         ----------
#         x : array-like
#             X coordinates of the data points.
#         y : array-like
#             Y coordinates of the data points.
#         yerr : array-like, optional
#             Error values for y-axis (symmetric or asymmetric).
#         xerr : array-like, optional
#             Error values for x-axis (symmetric or asymmetric).
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.errorbar`.
# 
#         Returns
#         -------
#         ErrorbarContainer
#             Container with the plotted errorbar lines.
# 
#         See Also
#         --------
#         stx_mean_std : Mean line with standard deviation shading.
#         stx_mean_ci : Mean line with confidence interval shading.
# 
#         Examples
#         --------
#         >>> ax.stx_errorbar(x, y, yerr=std, fmt='o-')
#         """
#         method_name = "stx_errorbar"
# 
#         if kwargs.get("label"):
#             n_samples = len(x)
#             kwargs["label"] = f"{kwargs['label']} ($n$={n_samples})"
# 
#         with self._no_tracking():
#             result = self._axis_mpl.errorbar(x, y, yerr=yerr, xerr=xerr, **kwargs)
# 
#         df_dict = {"x": x, "y": y}
#         if yerr is not None:
#             df_dict["yerr"] = yerr
#         if xerr is not None:
#             df_dict["xerr"] = xerr
#         tracked_dict = {"errorbar_df": pd.DataFrame(df_dict)}
#         self._track(track, id, method_name, tracked_dict, None)
# 
#         from scitex.plt.ax import style_errorbar
# 
#         style_errorbar(result)
# 
#         self._apply_scitex_postprocess(method_name, result)
# 
#         return result
# 
#     def stx_fill_between(
#         self,
#         x: ArrayLike,
#         y1: ArrayLike,
#         y2: Union[float, ArrayLike] = 0,
#         *,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ):
#         """Fill the area between two curves with SciTeX styling and tracking.
# 
#         Parameters
#         ----------
#         x : array-like
#             X coordinates for the fill region.
#         y1 : array-like
#             First y-boundary curve.
#         y2 : float or array-like, default 0
#             Second y-boundary curve.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.fill_between`.
# 
#         Returns
#         -------
#         PolyCollection
#             Collection representing the filled area.
# 
#         See Also
#         --------
#         stx_shaded_line : Line plot with shaded confidence region.
# 
#         Examples
#         --------
#         >>> ax.stx_fill_between(x, y_lower, y_upper, alpha=0.3)
#         """
#         method_name = "stx_fill_between"
# 
#         with self._no_tracking():
#             result = self._axis_mpl.fill_between(x, y1, y2, **kwargs)
# 
#         tracked_dict = {
#             "fill_between_df": pd.DataFrame(
#                 {
#                     "x": x,
#                     "y1": y1,
#                     "y2": y2 if hasattr(y2, "__len__") else [y2] * len(x),
#                 }
#             )
#         }
#         self._track(track, id, method_name, tracked_dict, None)
#         self._apply_scitex_postprocess(method_name, result)
# 
#         return result
# 
#     def stx_contour(
#         self,
#         X: ArrayLike,
#         Y: ArrayLike,
#         Z: ArrayLike,
#         *,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> QuadContourSet:
#         """Create a contour plot with SciTeX styling and tracking.
# 
#         Parameters
#         ----------
#         X : array-like
#             X coordinates of the grid (2D array or 1D for meshgrid).
#         Y : array-like
#             Y coordinates of the grid (2D array or 1D for meshgrid).
#         Z : array-like
#             Values at each grid point (2D array).
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.contour`.
# 
#         Returns
#         -------
#         QuadContourSet
#             The contour set object.
# 
#         See Also
#         --------
#         stx_imshow : Display data as an image.
#         mpl_contour : Raw matplotlib contour without styling.
# 
#         Examples
#         --------
#         >>> ax.stx_contour(X, Y, Z, levels=10)
#         """
#         method_name = "stx_contour"
# 
#         with self._no_tracking():
#             result = self._axis_mpl.contour(X, Y, Z, **kwargs)
# 
#         tracked_dict = {
#             "contour_df": pd.DataFrame(
#                 {"X": np.ravel(X), "Y": np.ravel(Y), "Z": np.ravel(Z)}
#             )
#         }
#         self._track(track, id, method_name, tracked_dict, None)
# 
#         self._apply_scitex_postprocess(method_name, result)
# 
#         return result
# 
#     def stx_imshow(
#         self,
#         data: ArrayLike,
#         *,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> AxesImage:
#         """Display data as an image with SciTeX styling and tracking.
# 
#         Parameters
#         ----------
#         data : array-like
#             Image data (2D or 3D array for RGB/RGBA).
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.imshow`.
# 
#         Returns
#         -------
#         AxesImage
#             The image object.
# 
#         See Also
#         --------
#         stx_image : Scientific image display with colorbar.
#         mpl_imshow : Raw matplotlib imshow without styling.
# 
#         Examples
#         --------
#         >>> ax.stx_imshow(image_array, cmap='viridis')
#         """
#         method_name = "stx_imshow"
# 
#         with self._no_tracking():
#             result = self._axis_mpl.imshow(data, **kwargs)
# 
#         if hasattr(data, "shape") and len(data.shape) == 2:
#             n_rows, n_cols = data.shape
#             df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])
#         else:
#             df = pd.DataFrame(data)
#         tracked_dict = {"imshow_df": df}
#         self._track(track, id, method_name, tracked_dict, None)
# 
#         self._apply_scitex_postprocess(method_name, result)
# 
#         return result
# 
#     def stx_boxplot(
#         self,
#         data: Union[ArrayLike, Sequence[ArrayLike]],
#         *,
#         colors: Optional[List[str]] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> dict:
#         """Create a boxplot with SciTeX styling and tracking.
# 
#         This is an alias for :meth:`stx_box`.
# 
#         Parameters
#         ----------
#         data : array-like or sequence of array-like
#             Data for the boxplot. Can be a single array or list of arrays.
#         colors : list of str, optional
#             Colors for each box.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to `matplotlib.axes.Axes.boxplot`.
# 
#         Returns
#         -------
#         dict
#             Dictionary mapping component names to artists.
# 
#         See Also
#         --------
#         stx_box : Primary boxplot method.
#         sns_boxplot : DataFrame-based boxplot.
#         stx_violin : Violin plot alternative.
# 
#         Examples
#         --------
#         >>> ax.stx_boxplot([data1, data2, data3], labels=['A', 'B', 'C'])
#         """
#         return self.stx_box(data, colors=colors, track=track, id=id, **kwargs)
# 
#     def stx_violinplot(
#         self,
#         data: Union[ArrayLike, Sequence[ArrayLike]],
#         *,
#         colors: Optional[List[str]] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> "Axes":
#         """Create a violin plot with SciTeX styling and tracking.
# 
#         This is an alias for :meth:`stx_violin`.
# 
#         Parameters
#         ----------
#         data : array-like or sequence of array-like
#             Data for the violin plot. Can be a single array or list of arrays.
#         colors : list of str, optional
#             Colors for each violin.
#         track : bool, default True
#             Enable data tracking for reproducibility.
#         id : str, optional
#             Unique identifier for this plot element.
#         **kwargs
#             Additional arguments passed to the violin plot function.
# 
#         Returns
#         -------
#         Axes
#             The axes with the violin plot.
# 
#         See Also
#         --------
#         stx_violin : Primary violin plot method.
#         sns_violinplot : DataFrame-based violin plot.
#         stx_box : Boxplot alternative.
# 
#         Examples
#         --------
#         >>> ax.stx_violinplot([data1, data2], labels=['A', 'B'])
#         """
#         return self.stx_violin(data, colors=colors, track=track, id=id, **kwargs)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_MatplotlibPlotMixin/_stx_aliases.py
# --------------------------------------------------------------------------------
