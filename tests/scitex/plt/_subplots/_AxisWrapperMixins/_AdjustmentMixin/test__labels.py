# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_AdjustmentMixin/_labels.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: _labels.py - Label rotation and legend handling
# 
# """Mixin for label rotation and legend positioning."""
# 
# import os
# 
# from scitex import logging
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# logger = logging.getLogger(__name__)
# 
# 
# class LabelsMixin:
#     """Mixin for label rotation and legend positioning."""
# 
#     def _get_ax_module(self):
#         """Lazy import ax module to avoid circular imports."""
#         from .....plt import ax as ax_module
#         return ax_module
# 
#     def rotate_labels(
#         self,
#         x: float = None,
#         y: float = None,
#         x_ha: str = None,
#         y_ha: str = None,
#         x_va: str = None,
#         y_va: str = None,
#         auto_adjust: bool = True,
#         scientific_convention: bool = True,
#         tight_layout: bool = False,
#     ) -> None:
#         """Rotate x and y axis labels with automatic positioning.
# 
#         Parameters
#         ----------
#         x : float or None, optional
#             Rotation angle for x-axis labels in degrees.
#         y : float or None, optional
#             Rotation angle for y-axis labels in degrees.
#         x_ha, y_ha : str or None, optional
#             Horizontal alignment for x/y-axis labels.
#         x_va, y_va : str or None, optional
#             Vertical alignment for x/y-axis labels.
#         auto_adjust : bool, optional
#             Whether to automatically adjust alignment. Default is True.
#         scientific_convention : bool, optional
#             Whether to follow scientific conventions. Default is True.
#         tight_layout : bool, optional
#             Whether to apply tight_layout. Default is False.
#         """
#         self._axis_mpl = self._get_ax_module().rotate_labels(
#             self._axis_mpl,
#             x=x,
#             y=y,
#             x_ha=x_ha,
#             y_ha=y_ha,
#             x_va=x_va,
#             y_va=y_va,
#             auto_adjust=auto_adjust,
#             scientific_convention=scientific_convention,
#             tight_layout=tight_layout,
#         )
# 
#     def legend(
#         self, *args, loc: str = "best", check_overlap: bool = False, **kwargs
#     ) -> None:
#         """Places legend at specified location, with support for outside positions.
# 
#         Parameters
#         ----------
#         *args : tuple
#             Positional arguments (handles, labels) as in matplotlib
#         loc : str
#             Legend position. Default is "best" (matplotlib auto-placement).
#             Special positions:
#             - "best": Matplotlib automatic placement
#             - "outer": Place outside plot area (right side)
#             - "separate": Save legend as a separate figure file
#             - upper/lower/center variants: e.g. "upper right out"
#         check_overlap : bool
#             If True, checks for overlap between legend and data.
#         **kwargs : dict
#             Additional keyword arguments passed to legend()
#         """
#         import matplotlib.pyplot as plt
# 
#         if loc == "outer":
#             legend = self._axis_mpl.legend(
#                 *args, loc="center left", bbox_to_anchor=(1.02, 0.5), **kwargs
#             )
#             if hasattr(self, "_figure_wrapper") and self._figure_wrapper:
#                 self._figure_wrapper._fig_mpl.tight_layout()
#                 self._figure_wrapper._fig_mpl.subplots_adjust(right=0.85)
#             return legend
# 
#         elif loc == "separate":
#             handles, labels = self._axis_mpl.get_legend_handles_labels()
#             if not handles:
#                 logger.warning("No legend handles found.")
#                 return None
# 
#             fig = self._axis_mpl.get_figure()
#             if not hasattr(fig, "_separate_legend_params"):
#                 fig._separate_legend_params = []
# 
#             figsize = kwargs.pop("figsize", (4, 3))
#             dpi = kwargs.pop("dpi", 150)
#             frameon = kwargs.pop("frameon", True)
#             fancybox = kwargs.pop("fancybox", True)
#             shadow = kwargs.pop("shadow", True)
# 
#             axis_id = self._get_axis_id(fig)
# 
#             fig._separate_legend_params.append({
#                 "axis": self._axis_mpl,
#                 "axis_id": axis_id,
#                 "handles": handles,
#                 "labels": labels,
#                 "figsize": figsize,
#                 "dpi": dpi,
#                 "frameon": frameon,
#                 "fancybox": fancybox,
#                 "shadow": shadow,
#                 "kwargs": kwargs,
#             })
# 
#             if self._axis_mpl.get_legend():
#                 self._axis_mpl.get_legend().remove()
# 
#             return None
# 
#         outside_positions = {
#             "upper right out": ("center left", (1.15, 0.85)),
#             "right upper out": ("center left", (1.15, 0.85)),
#             "center right out": ("center left", (1.15, 0.5)),
#             "right out": ("center left", (1.15, 0.5)),
#             "right": ("center left", (1.05, 0.5)),
#             "lower right out": ("center left", (1.15, 0.15)),
#             "right lower out": ("center left", (1.15, 0.15)),
#             "upper left out": ("center right", (-0.25, 0.85)),
#             "left upper out": ("center right", (-0.25, 0.85)),
#             "center left out": ("center right", (-0.25, 0.5)),
#             "left out": ("center right", (-0.25, 0.5)),
#             "left": ("center right", (-0.15, 0.5)),
#             "lower left out": ("center right", (-0.25, 0.15)),
#             "left lower out": ("center right", (-0.25, 0.15)),
#             "upper center out": ("lower center", (0.5, 1.25)),
#             "upper out": ("lower center", (0.5, 1.25)),
#             "lower center out": ("upper center", (0.5, -0.25)),
#             "lower out": ("upper center", (0.5, -0.25)),
#         }
# 
#         if loc in outside_positions:
#             location, bbox = outside_positions[loc]
#             legend_obj = self._axis_mpl.legend(
#                 *args, loc=location, bbox_to_anchor=bbox, **kwargs
#             )
#         else:
#             legend_obj = self._axis_mpl.legend(*args, loc=loc, **kwargs)
# 
#         if check_overlap and legend_obj is not None:
#             self._check_legend_overlap(legend_obj)
# 
#         return legend_obj
# 
#     def _get_axis_id(self, fig):
#         """Get unique axis identifier for separate legend handling."""
#         axis_id = None
# 
#         try:
#             fig_axes = fig.get_axes()
#             for idx, ax in enumerate(fig_axes):
#                 if ax is self._axis_mpl:
#                     axis_id = f"ax_{idx:02d}"
#                     break
#         except:
#             pass
# 
#         if axis_id is None and hasattr(self._axis_mpl, "get_subplotspec"):
#             try:
#                 spec = self._axis_mpl.get_subplotspec()
#                 if spec is not None:
#                     gridspec = spec.get_gridspec()
#                     nrows, ncols = gridspec.get_geometry()
#                     rowspan = spec.rowspan
#                     colspan = spec.colspan
#                     row_start = rowspan.start if hasattr(rowspan, "start") else rowspan
#                     col_start = colspan.start if hasattr(colspan, "start") else colspan
#                     flat_idx = row_start * ncols + col_start
#                     axis_id = f"ax_{flat_idx:02d}"
#             except:
#                 pass
# 
#         if axis_id is None:
#             axis_id = f"ax_{len(fig._separate_legend_params):02d}"
# 
#         return axis_id
# 
#     def _check_legend_overlap(self, legend_obj):
#         """Check if legend overlaps with plotted data and issue warning if needed."""
#         import warnings
#         import matplotlib.transforms as transforms
#         import numpy as np
# 
#         try:
#             fig = self._axis_mpl.get_figure()
#             fig.canvas.draw()
# 
#             legend_bbox = legend_obj.get_window_extent(fig.canvas.get_renderer())
#             inv_transform = self._axis_mpl.transData.inverted()
#             legend_bbox_data = legend_bbox.transformed(inv_transform)
# 
#             data_bboxes = []
# 
#             for line in self._axis_mpl.get_lines():
#                 if line.get_visible():
#                     try:
#                         data = line.get_xydata()
#                         if len(data) > 0:
#                             data_bboxes.append(data)
#                     except:
#                         pass
# 
#             for collection in self._axis_mpl.collections:
#                 if collection.get_visible():
#                     try:
#                         offsets = collection.get_offsets()
#                         if len(offsets) > 0:
#                             data_bboxes.append(offsets)
#                     except:
#                         pass
# 
#             if data_bboxes:
#                 all_data = np.vstack(data_bboxes)
# 
#                 x_overlap = (all_data[:, 0] >= legend_bbox_data.x0) & (
#                     all_data[:, 0] <= legend_bbox_data.x1
#                 )
#                 y_overlap = (all_data[:, 1] >= legend_bbox_data.y0) & (
#                     all_data[:, 1] <= legend_bbox_data.y1
#                 )
#                 overlap_points = np.sum(x_overlap & y_overlap)
#                 overlap_pct = (overlap_points / len(all_data)) * 100
# 
#                 if overlap_pct > 5:
#                     logger.warning(
#                         f"Legend overlaps with {overlap_pct:.1f}% of data points. "
#                         f"Consider using loc='outer' or loc='separate'."
#                     )
#                     return True
# 
#         except Exception:
#             pass
# 
#         return False
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_AdjustmentMixin/_labels.py
# --------------------------------------------------------------------------------
