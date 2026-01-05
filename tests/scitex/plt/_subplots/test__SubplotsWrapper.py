#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-29 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/tests/scitex/plt/_subplots/test__SubplotsWrapper.py

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

import scitex


class TestSubplotsWrapper:
    """Test cases for scitex.plt.subplots wrapper functionality."""

    def test_single_axis(self):
        """Test that single axis returns an AxisWrapper object."""
        fig, ax = scitex.plt.subplots()
        assert hasattr(ax, "plot"), "Single axis should have plot method"
        assert hasattr(ax, "export_as_csv"), "Should have export_as_csv method"
        scitex.plt.close(fig)

    def test_1d_array_single_row(self):
        """Test that single row multiple columns returns 1D array."""
        fig, axes = scitex.plt.subplots(1, 3)
        assert hasattr(axes, "__len__"), "Should return array-like object"
        assert len(axes) == 3, "Should have 3 axes"
        # Test individual axis access
        for i in range(3):
            assert hasattr(axes[i], "plot"), f"axes[{i}] should have plot method"
        scitex.plt.close(fig)

    def test_1d_array_single_column(self):
        """Test that multiple rows single column returns 1D array."""
        fig, axes = scitex.plt.subplots(3, 1)
        assert hasattr(axes, "__len__"), "Should return array-like object"
        assert len(axes) == 3, "Should have 3 axes"
        # Test individual axis access
        for i in range(3):
            assert hasattr(axes[i], "plot"), f"axes[{i}] should have plot method"
        scitex.plt.close(fig)

    def test_2d_array_indexing(self):
        """Test that 2D grid allows 2D indexing (the main bug fix)."""
        fig, axes = scitex.plt.subplots(4, 3)

        # Test shape property
        assert hasattr(axes, "shape"), "Should have shape property"
        assert axes.shape == (4, 3), "Shape should be (4, 3)"

        # Test 2D indexing - this is the core fix
        for row in range(4):
            for col in range(3):
                ax = axes[row, col]
                assert hasattr(
                    ax, "plot"
                ), f"axes[{row}, {col}] should have plot method"
                # Test that we can actually plot
                ax.plot([1, 2, 3], [1, 2, 3])

        scitex.plt.close(fig)

    def test_2d_array_row_access(self):
        """Test accessing entire rows from 2D array."""
        fig, axes = scitex.plt.subplots(4, 3)

        # Access entire row
        row_axes = axes[0]  # First row
        assert len(row_axes) == 3, "Row should have 3 axes"

        # Each element in row should be plottable
        for i, ax in enumerate(row_axes):
            assert hasattr(ax, "plot"), f"Row axis [{i}] should have plot method"

        scitex.plt.close(fig)

    def test_2d_array_slice_access(self):
        """Test slice access on 2D array."""
        fig, axes = scitex.plt.subplots(4, 3)

        # Access slice of rows
        slice_axes = axes[1:3]  # Rows 1 and 2
        assert hasattr(slice_axes, "shape"), "Slice should return AxesWrapper"
        assert slice_axes.shape == (2, 3), "Slice shape should be (2, 3)"

        scitex.plt.close(fig)

    def test_backward_compatibility_flat_iteration(self):
        """Test that flat iteration still works for backward compatibility."""
        fig, axes = scitex.plt.subplots(4, 3)

        # Test iteration (should be flattened)
        ax_list = list(axes)
        assert len(ax_list) == 12, "Iteration should yield 12 axes (flattened)"

        # Test each axis is plottable
        for i, ax in enumerate(axes):
            assert hasattr(ax, "plot"), f"Iterated axis {i} should have plot method"

        scitex.plt.close(fig)

    def test_export_as_csv_multi_axes(self):
        """Test export_as_csv functionality with multiple axes."""
        fig, axes = scitex.plt.subplots(2, 2)

        # Plot on each axis
        axes[0, 0].plot([1, 2, 3], [1, 2, 3], id="plot00")
        axes[0, 1].plot([1, 2, 3], [3, 2, 1], id="plot01")
        axes[1, 0].plot([1, 2, 3], [2, 3, 4], id="plot10")
        axes[1, 1].plot([1, 2, 3], [4, 3, 2], id="plot11")

        # Test export functionality
        df = axes.export_as_csv()
        assert df is not None, "Should return a DataFrame"
        assert len(df.columns) > 0, "DataFrame should have columns"

        scitex.plt.close(fig)

    def test_matplotlib_compatibility(self):
        """Test that the behavior matches matplotlib's for common use cases."""
        # Compare with matplotlib behavior
        mpl_fig, mpl_axes = plt.subplots(3, 2)
        scitex_fig, scitex_axes = scitex.plt.subplots(3, 2)

        # Both should have the same shape
        assert scitex_axes.shape == mpl_axes.shape, "Should have same shape as matplotlib"

        # Both should allow 2D indexing
        for i in range(3):
            for j in range(2):
                # This should not raise an error
                mpl_ax = mpl_axes[i, j]
                scitex_ax = scitex_axes[i, j]
                assert hasattr(scitex_ax, "plot"), "scitex axis should have plot method"

        plt.close(mpl_fig)
        scitex.plt.close(scitex_fig)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_SubplotsWrapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-29 03:46:53 (ywatanabe)"
# # File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/scitex_repo/src/scitex/plt/_subplots/_SubplotsWrapper.py
# # ----------------------------------------
# import os
# 
# __FILE__ = "./src/scitex/plt/_subplots/_SubplotsWrapper.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from collections import OrderedDict
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# from ._AxesWrapper import AxesWrapper
# from ._AxisWrapper import AxisWrapper
# from ._FigWrapper import FigWrapper
# 
# # Register Arial fonts at module import
# import matplotlib.font_manager as fm
# import matplotlib as mpl
# import os
# 
# _arial_enabled = False
# 
# # Try to find Arial
# try:
#     fm.findfont("Arial", fallback_to_default=False)
#     _arial_enabled = True
# except Exception:
#     # Search for Arial font files and register them
#     arial_paths = [
#         f
#         for f in fm.findSystemFonts()
#         if os.path.basename(f).lower().startswith("arial")
#     ]
# 
#     if arial_paths:
#         for path in arial_paths:
#             try:
#                 fm.fontManager.addfont(path)
#             except Exception:
#                 pass
# 
#         # Verify Arial is now available
#         try:
#             fm.findfont("Arial", fallback_to_default=False)
#             _arial_enabled = True
#         except Exception:
#             pass
# 
# # Configure matplotlib to use Arial if available
# if _arial_enabled:
#     mpl.rcParams["font.family"] = "Arial"
#     mpl.rcParams["font.sans-serif"] = [
#         "Arial",
#         "Helvetica",
#         "DejaVu Sans",
#         "Liberation Sans",
#     ]
# else:
#     # Warn about missing Arial
#     from scitex import logging as _logging
# 
#     _logger = _logging.getLogger(__name__)
#     _logger.warning(
#         "Arial font not found. Using fallback fonts (Helvetica/DejaVu Sans). "
#         "For publication figures with Arial: sudo apt-get install ttf-mscorefonts-installer && fc-cache -fv"
#     )
# 
# 
# class SubplotsWrapper:
#     """
#     A wrapper class monitors data plotted using the ax methods from matplotlib.pyplot.
#     This data can be converted into a CSV file formatted for SigmaPlot compatibility.
#     """
# 
#     def __init__(self):
#         self._subplots_wrapper_history = OrderedDict()
#         self._fig_scitex = None
#         self._counter_part = plt.subplots
# 
#     def __call__(
#         self,
#         *args,
#         track=True,
#         sharex=False,
#         sharey=False,
#         constrained_layout=None,
#         # MM-control parameters (unified style system)
#         axes_width_mm=None,
#         axes_height_mm=None,
#         margin_left_mm=None,
#         margin_right_mm=None,
#         margin_bottom_mm=None,
#         margin_top_mm=None,
#         space_w_mm=None,
#         space_h_mm=None,
#         axes_thickness_mm=None,
#         tick_length_mm=None,
#         tick_thickness_mm=None,
#         trace_thickness_mm=None,
#         marker_size_mm=None,
#         axis_font_size_pt=None,
#         tick_font_size_pt=None,
#         title_font_size_pt=None,
#         legend_font_size_pt=None,
#         suptitle_font_size_pt=None,
#         n_ticks=None,
#         mode=None,
#         dpi=None,
#         styles=None,  # List of style dicts for per-axes control
#         transparent=None,  # Transparent background (default: from SCITEX_STYLE.yaml)
#         theme=None,  # Color theme: "light" or "dark" (default: "light")
#         **kwargs,
#     ):
#         """
#         Create figure and axes with optional millimeter-based control.
# 
#         Parameters
#         ----------
#         *args : int
#             nrows, ncols passed to matplotlib.pyplot.subplots
#         track : bool, optional
#             Track plotting operations for CSV export (default: True)
#         sharex, sharey : bool, optional
#             Share axes (default: False)
#         constrained_layout : dict or None, optional
#             Layout engine parameters
# 
#         MM-Control Parameters (Unified Style System)
#         ---------------------------------------------
#         axes_width_mm : float or list, optional
#             Axes width in mm (single value for all, or list for each)
#         axes_height_mm : float or list, optional
#             Axes height in mm (single value for all, or list for each)
#         margin_left_mm : float, optional
#             Left margin in mm (default: 5.0)
#         margin_right_mm : float, optional
#             Right margin in mm (default: 2.0)
#         margin_bottom_mm : float, optional
#             Bottom margin in mm (default: 5.0)
#         margin_top_mm : float, optional
#             Top margin in mm (default: 2.0)
#         space_w_mm : float, optional
#             Horizontal spacing between axes in mm (default: 3.0)
#         space_h_mm : float, optional
#             Vertical spacing between axes in mm (default: 3.0)
#         axes_thickness_mm : float, optional
#             Axes spine thickness in mm
#         tick_length_mm : float, optional
#             Tick length in mm
#         tick_thickness_mm : float, optional
#             Tick thickness in mm
#         trace_thickness_mm : float, optional
#             Plot line thickness in mm
#         axis_font_size_pt : float, optional
#             Axis label font size in points
#         tick_font_size_pt : float, optional
#             Tick label font size in points
#         mode : str, optional
#             'publication' or 'display' (default: None, uses standard matplotlib)
#         dpi : int, optional
#             Resolution (default: 300 for publication, 100 for display)
#         styles : list of dict, optional
#             Individual style dicts for each axes
#         transparent : bool, optional
#             Create figure with transparent background (default: False)
#             Useful for publication-ready figures that will be cropped
#         theme : str, optional
#             Color theme: "light" or "dark" (default: "light")
#             Dark mode uses eye-friendly colors optimized for dark backgrounds
#         **kwargs
#             Additional arguments passed to matplotlib.pyplot.subplots
# 
#         Returns
#         -------
#         fig : FigWrapper
#             Wrapped matplotlib Figure
#         ax or axes : AxisWrapper or AxesWrapper
#             Wrapped matplotlib Axes
# 
#         Examples
#         --------
#         Single axes with style:
# 
#         >>> fig, ax = stx.plt.subplots(
#         ...     axes_width_mm=30,
#         ...     axes_height_mm=21,
#         ...     axes_thickness_mm=0.2,
#         ...     tick_length_mm=0.8,
#         ...     mode='publication'
#         ... )
# 
#         Multiple axes with uniform style:
# 
#         >>> fig, axes = stx.plt.subplots(
#         ...     nrows=2, ncols=3,
#         ...     axes_width_mm=30,
#         ...     axes_height_mm=21,
#         ...     space_w_mm=3,
#         ...     space_h_mm=3,
#         ...     mode='publication'
#         ... )
# 
#         Using style preset:
# 
#         >>> NATURE_STYLE = {
#         ...     'axes_width_mm': 30,
#         ...     'axes_height_mm': 21,
#         ...     'axes_thickness_mm': 0.2,
#         ...     'tick_length_mm': 0.8,
#         ... }
#         >>> fig, ax = stx.plt.subplots(**NATURE_STYLE)
#         """
# 
#         # Use resolve_style_value for priority: direct → yaml → env → default
#         from scitex.plt.styles import (
#             resolve_style_value as _resolve,
#             SCITEX_STYLE as _S,
#         )
# 
#         # Resolve all style values with proper priority chain
#         axes_width_mm = _resolve(
#             "axes.width_mm", axes_width_mm, _S.get("axes_width_mm")
#         )
#         axes_height_mm = _resolve(
#             "axes.height_mm", axes_height_mm, _S.get("axes_height_mm")
#         )
#         margin_left_mm = _resolve(
#             "margins.left_mm", margin_left_mm, _S.get("margin_left_mm")
#         )
#         margin_right_mm = _resolve(
#             "margins.right_mm", margin_right_mm, _S.get("margin_right_mm")
#         )
#         margin_bottom_mm = _resolve(
#             "margins.bottom_mm", margin_bottom_mm, _S.get("margin_bottom_mm")
#         )
#         margin_top_mm = _resolve(
#             "margins.top_mm", margin_top_mm, _S.get("margin_top_mm")
#         )
#         space_w_mm = _resolve("spacing.horizontal_mm", space_w_mm, _S.get("space_w_mm"))
#         space_h_mm = _resolve("spacing.vertical_mm", space_h_mm, _S.get("space_h_mm"))
#         axes_thickness_mm = _resolve(
#             "axes.thickness_mm", axes_thickness_mm, _S.get("axes_thickness_mm")
#         )
#         tick_length_mm = _resolve(
#             "ticks.length_mm", tick_length_mm, _S.get("tick_length_mm")
#         )
#         tick_thickness_mm = _resolve(
#             "ticks.thickness_mm", tick_thickness_mm, _S.get("tick_thickness_mm")
#         )
#         trace_thickness_mm = _resolve(
#             "lines.trace_mm", trace_thickness_mm, _S.get("trace_thickness_mm")
#         )
#         marker_size_mm = _resolve(
#             "markers.size_mm", marker_size_mm, _S.get("marker_size_mm")
#         )
#         axis_font_size_pt = _resolve(
#             "fonts.axis_label_pt", axis_font_size_pt, _S.get("axis_font_size_pt")
#         )
#         tick_font_size_pt = _resolve(
#             "fonts.tick_label_pt", tick_font_size_pt, _S.get("tick_font_size_pt")
#         )
#         title_font_size_pt = _resolve(
#             "fonts.title_pt", title_font_size_pt, _S.get("title_font_size_pt")
#         )
#         legend_font_size_pt = _resolve(
#             "fonts.legend_pt", legend_font_size_pt, _S.get("legend_font_size_pt")
#         )
#         suptitle_font_size_pt = _resolve(
#             "fonts.suptitle_pt", suptitle_font_size_pt, _S.get("suptitle_font_size_pt")
#         )
#         n_ticks = _resolve("ticks.n_ticks", n_ticks, _S.get("n_ticks"), int)
#         dpi = _resolve("output.dpi", dpi, _S.get("dpi"), int)
#         # Resolve transparent from YAML (default: True in SCITEX_STYLE.yaml)
#         if transparent is None:
#             transparent = _S.get("transparent", True)
#         if mode is None:
#             mode = _S.get("mode", "publication")
#         # Resolve theme from YAML (default: "light")
#         if theme is None:
#             theme = _resolve("theme.mode", None, "light", str)
# 
#         # Always use mm-control pathway with SCITEX_STYLE defaults
#         if True:
#             # Use mm-control pathway
#             return self._create_with_mm_control(
#                 *args,
#                 track=track,
#                 sharex=sharex,
#                 sharey=sharey,
#                 axes_width_mm=axes_width_mm,
#                 axes_height_mm=axes_height_mm,
#                 margin_left_mm=margin_left_mm,
#                 margin_right_mm=margin_right_mm,
#                 margin_bottom_mm=margin_bottom_mm,
#                 margin_top_mm=margin_top_mm,
#                 space_w_mm=space_w_mm,
#                 space_h_mm=space_h_mm,
#                 axes_thickness_mm=axes_thickness_mm,
#                 tick_length_mm=tick_length_mm,
#                 tick_thickness_mm=tick_thickness_mm,
#                 trace_thickness_mm=trace_thickness_mm,
#                 marker_size_mm=marker_size_mm,
#                 axis_font_size_pt=axis_font_size_pt,
#                 tick_font_size_pt=tick_font_size_pt,
#                 title_font_size_pt=title_font_size_pt,
#                 legend_font_size_pt=legend_font_size_pt,
#                 suptitle_font_size_pt=suptitle_font_size_pt,
#                 n_ticks=n_ticks,
#                 mode=mode,
#                 dpi=dpi,
#                 styles=styles,
#                 transparent=transparent,
#                 theme=theme,
#                 **kwargs,
#             )
# 
#     def _create_with_mm_control(
#         self,
#         *args,
#         track=True,
#         sharex=False,
#         sharey=False,
#         axes_width_mm=None,
#         axes_height_mm=None,
#         margin_left_mm=None,
#         margin_right_mm=None,
#         margin_bottom_mm=None,
#         margin_top_mm=None,
#         space_w_mm=None,
#         space_h_mm=None,
#         axes_thickness_mm=None,
#         tick_length_mm=None,
#         tick_thickness_mm=None,
#         trace_thickness_mm=None,
#         marker_size_mm=None,
#         axis_font_size_pt=None,
#         tick_font_size_pt=None,
#         title_font_size_pt=None,
#         legend_font_size_pt=None,
#         suptitle_font_size_pt=None,
#         label_pad_pt=None,
#         tick_pad_pt=None,
#         title_pad_pt=None,
#         font_family=None,
#         n_ticks=None,
#         mode=None,
#         dpi=None,
#         styles=None,
#         transparent=None,  # Resolved from caller
#         theme=None,  # Color theme: "light" or "dark"
#         **kwargs,
#     ):
#         """Create figure with mm-based control over axes dimensions."""
#         from scitex.plt.utils import mm_to_inch, apply_style_mm
# 
#         # Parse nrows, ncols from args or kwargs (like matplotlib.pyplot.subplots)
#         nrows, ncols = 1, 1
#         if len(args) >= 1:
#             nrows = args[0]
#         elif "nrows" in kwargs:
#             nrows = kwargs.pop("nrows")
#         if len(args) >= 2:
#             ncols = args[1]
#         elif "ncols" in kwargs:
#             ncols = kwargs.pop("ncols")
# 
#         n_axes = nrows * ncols
# 
#         # Apply mode-specific defaults
#         if mode == "display":
#             scale_factor = 3.0
#             dpi = dpi or 100
#         else:  # publication or None
#             scale_factor = 1.0
#             dpi = dpi or 300
# 
#         # Set defaults - if value is provided, apply scaling; if not, use scaled default
#         if axes_width_mm is None:
#             axes_width_mm = 30.0 * scale_factor
#         elif mode == "display":
#             axes_width_mm = axes_width_mm * scale_factor
# 
#         if axes_height_mm is None:
#             axes_height_mm = 21.0 * scale_factor
#         elif mode == "display":
#             axes_height_mm = axes_height_mm * scale_factor
# 
#         margin_left_mm = (
#             margin_left_mm if margin_left_mm is not None else (5.0 * scale_factor)
#         )
#         margin_right_mm = (
#             margin_right_mm if margin_right_mm is not None else (2.0 * scale_factor)
#         )
#         margin_bottom_mm = (
#             margin_bottom_mm if margin_bottom_mm is not None else (5.0 * scale_factor)
#         )
#         margin_top_mm = (
#             margin_top_mm if margin_top_mm is not None else (2.0 * scale_factor)
#         )
#         space_w_mm = space_w_mm if space_w_mm is not None else (3.0 * scale_factor)
#         space_h_mm = space_h_mm if space_h_mm is not None else (3.0 * scale_factor)
# 
#         # Handle list vs scalar for axes_width_mm and axes_height_mm
#         if isinstance(axes_width_mm, (list, tuple)):
#             ax_widths_mm = list(axes_width_mm)
#             if len(ax_widths_mm) != n_axes:
#                 raise ValueError(
#                     f"axes_width_mm list length ({len(ax_widths_mm)}) must match nrows*ncols ({n_axes})"
#                 )
#         else:
#             ax_widths_mm = [axes_width_mm] * n_axes
# 
#         if isinstance(axes_height_mm, (list, tuple)):
#             ax_heights_mm = list(axes_height_mm)
#             if len(ax_heights_mm) != n_axes:
#                 raise ValueError(
#                     f"axes_height_mm list length ({len(ax_heights_mm)}) must match nrows*ncols ({n_axes})"
#                 )
#         else:
#             ax_heights_mm = [axes_height_mm] * n_axes
# 
#         # Calculate figure size from axes grid
#         # For simplicity, use max width per column and max height per row
#         ax_widths_2d = np.array(ax_widths_mm).reshape(nrows, ncols)
#         ax_heights_2d = np.array(ax_heights_mm).reshape(nrows, ncols)
# 
#         max_widths_per_col = ax_widths_2d.max(axis=0)  # Max width in each column
#         max_heights_per_row = ax_heights_2d.max(axis=1)  # Max height in each row
# 
#         total_width_mm = (
#             margin_left_mm
#             + max_widths_per_col.sum()
#             + (ncols - 1) * space_w_mm
#             + margin_right_mm
#         )
#         total_height_mm = (
#             margin_bottom_mm
#             + max_heights_per_row.sum()
#             + (nrows - 1) * space_h_mm
#             + margin_top_mm
#         )
# 
#         # Create figure with calculated size
#         figsize_inch = (mm_to_inch(total_width_mm), mm_to_inch(total_height_mm))
#         if transparent:
#             # Transparent background for publication figures
#             self._fig_mpl = plt.figure(figsize=figsize_inch, dpi=dpi, facecolor="none")
#         else:
#             self._fig_mpl = plt.figure(figsize=figsize_inch, dpi=dpi)
# 
#         # Store theme on figure for later retrieval (e.g., when saving plot.json)
#         if theme is not None:
#             self._fig_mpl._scitex_theme = theme
# 
#         # Create axes array and position each one manually
#         axes_mpl_list = []
#         ax_idx = 0
# 
#         for row in range(nrows):
#             for col in range(ncols):
#                 # Calculate position for this axes
#                 # Left position: left margin + sum of previous column widths + spacing
#                 left_mm = (
#                     margin_left_mm + max_widths_per_col[:col].sum() + col * space_w_mm
#                 )
# 
#                 # Bottom position: bottom margin + sum of heights above this row + spacing
#                 # (rows are counted from top in matplotlib)
#                 rows_below = nrows - row - 1
#                 bottom_mm = (
#                     margin_bottom_mm
#                     + max_heights_per_row[row + 1 :].sum()
#                     + rows_below * space_h_mm
#                 )
# 
#                 # Convert to figure coordinates [0-1]
#                 left = left_mm / total_width_mm
#                 bottom = bottom_mm / total_height_mm
#                 width = ax_widths_mm[ax_idx] / total_width_mm
#                 height = ax_heights_mm[ax_idx] / total_height_mm
# 
#                 # Create axes at exact position with transparent background
#                 ax_mpl = self._fig_mpl.add_axes([left, bottom, width, height])
#                 if transparent:
#                     ax_mpl.patch.set_alpha(0.0)  # Make axes background transparent
#                 axes_mpl_list.append(ax_mpl)
# 
#                 # Tag with metadata
#                 ax_mpl._scitex_metadata = {
#                     "created_with": "scitex.plt.subplots",
#                     "mode": mode or "publication",
#                     "axes_size_mm": (ax_widths_mm[ax_idx], ax_heights_mm[ax_idx]),
#                     "position_in_grid": (row, col),
#                 }
# 
#                 ax_idx += 1
# 
#         # Apply styling to each axes
#         suptitle_font_size_pt = None
#         for i, ax_mpl in enumerate(axes_mpl_list):
#             # Determine which style dict to use
#             if styles is not None:
#                 if isinstance(styles, list):
#                     if len(styles) != n_axes:
#                         raise ValueError(
#                             f"styles list length ({len(styles)}) must match nrows*ncols ({n_axes})"
#                         )
#                     style_dict = styles[i]
#                 else:
#                     style_dict = styles
#             else:
#                 # Build style dict from individual parameters
#                 style_dict = {}
#                 if axes_thickness_mm is not None:
#                     style_dict["axis_thickness_mm"] = axes_thickness_mm
#                 if tick_length_mm is not None:
#                     style_dict["tick_length_mm"] = tick_length_mm
#                 if tick_thickness_mm is not None:
#                     style_dict["tick_thickness_mm"] = tick_thickness_mm
#                 if trace_thickness_mm is not None:
#                     style_dict["trace_thickness_mm"] = trace_thickness_mm
#                 if marker_size_mm is not None:
#                     style_dict["marker_size_mm"] = marker_size_mm
#                 if axis_font_size_pt is not None:
#                     style_dict["axis_font_size_pt"] = axis_font_size_pt
#                 if tick_font_size_pt is not None:
#                     style_dict["tick_font_size_pt"] = tick_font_size_pt
#                 if title_font_size_pt is not None:
#                     style_dict["title_font_size_pt"] = title_font_size_pt
#                 if legend_font_size_pt is not None:
#                     style_dict["legend_font_size_pt"] = legend_font_size_pt
#                 if suptitle_font_size_pt is not None:
#                     style_dict["suptitle_font_size_pt"] = suptitle_font_size_pt
#                 if label_pad_pt is not None:
#                     style_dict["label_pad_pt"] = label_pad_pt
#                 if tick_pad_pt is not None:
#                     style_dict["tick_pad_pt"] = tick_pad_pt
#                 if title_pad_pt is not None:
#                     style_dict["title_pad_pt"] = title_pad_pt
#                 if font_family is not None:
#                     style_dict["font_family"] = font_family
#                 if n_ticks is not None:
#                     style_dict["n_ticks"] = n_ticks
# 
#             # Always add theme to style_dict (default: "light")
#             if theme is not None:
#                 style_dict["theme"] = theme
# 
#             # Extract suptitle font size if available
#             if "suptitle_font_size_pt" in style_dict:
#                 suptitle_font_size_pt_value = style_dict["suptitle_font_size_pt"]
#             else:
#                 suptitle_font_size_pt_value = None
# 
#             # Apply style if not empty
#             if style_dict:
#                 apply_style_mm(ax_mpl, style_dict)
#                 # Add style to metadata
#                 ax_mpl._scitex_metadata["style_mm"] = style_dict
# 
#         # Store suptitle font size in figure metadata for later use
#         if suptitle_font_size_pt_value is not None:
#             self._fig_mpl._scitex_suptitle_font_size_pt = suptitle_font_size_pt_value
# 
#         # Wrap the figure
#         self._fig_scitex = FigWrapper(self._fig_mpl)
# 
#         # Reshape axes list to match grid shape
#         axes_array_mpl = np.array(axes_mpl_list).reshape(nrows, ncols)
# 
#         # Handle single axis case
#         if n_axes == 1:
#             ax_mpl_scalar = axes_array_mpl.item()
#             self._axis_scitex = AxisWrapper(self._fig_scitex, ax_mpl_scalar, track)
#             # ALWAYS use list for consistency with matplotlib (fig.axes is always a list)
#             self._fig_scitex.axes = [self._axis_scitex]
#             # Store reference to scitex wrapper on matplotlib axes for metadata collection
#             ax_mpl_scalar._scitex_wrapper = self._axis_scitex
#             return self._fig_scitex, self._axis_scitex
# 
#         # Handle multiple axes case
#         axes_flat_scitex_list = []
#         for ax_mpl in axes_mpl_list:
#             ax_scitex = AxisWrapper(self._fig_scitex, ax_mpl, track)
#             # Store reference to scitex wrapper on matplotlib axes for metadata collection
#             ax_mpl._scitex_wrapper = ax_scitex
#             axes_flat_scitex_list.append(ax_scitex)
# 
#         axes_array_scitex = np.array(axes_flat_scitex_list).reshape(nrows, ncols)
#         self._axes_scitex = AxesWrapper(self._fig_scitex, axes_array_scitex)
#         self._fig_scitex.axes = self._axes_scitex
# 
#         return self._fig_scitex, self._axes_scitex
# 
#     # def __getattr__(self, name):
#     #     """
#     #     Fallback to fetch attributes from the original matplotlib.pyplot.subplots function
#     #     if they are not defined directly in this wrapper instance.
#     #     This allows accessing attributes like __name__, __doc__ etc. from the original function.
#     #     """
#     #     print(f"Attribute of SubplotsWrapper: {name}")
#     #     # Check if the attribute exists in the counterpart function
#     #     if hasattr(self._counter_part, name):
#     #         return getattr(self._counter_part, name)
#     #     # Raise the standard error if not found in the wrapper or the counterpart
#     #     raise AttributeError(
#     #         f"'{type(self).__name__}' object and its counterpart '{self._counter_part.__name__}' have no attribute '{name}'"
#     #     )
# 
#     def __dir__(
#         self,
#     ):
#         """
#         Provide combined directory for tab completion, including
#         attributes from this wrapper and the original matplotlib.pyplot.subplots function.
#         """
#         # Get attributes defined explicitly in this instance/class
#         local_attrs = set(super().__dir__())
#         # Get attributes from the counterpart function
#         try:
#             counterpart_attrs = set(dir(self._counter_part))
#         except Exception:
#             counterpart_attrs = set()
#         # Return the sorted union
#         return sorted(local_attrs.union(counterpart_attrs))
# 
# 
# # Instantiate the wrapper. This instance will be imported and used.
# subplots = SubplotsWrapper()
# 
# if __name__ == "__main__":
#     import matplotlib
#     import scitex
# 
#     matplotlib.use("TkAgg")  # "TkAgg"
# 
#     fig, ax = subplots()
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
#     scitex.io.save(fig, "/tmp/subplots_demo/plots.png")
# 
#     # Behaves like native matplotlib.pyplot.subplots without tracking
#     fig, ax = subplots(track=False)
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
#     scitex.io.save(fig, "/tmp/subplots_demo/plots.png")
# 
#     fig, ax = subplots()
#     ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
#     ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
#     scitex.io.save(fig, "/tmp/subplots_demo/scatters.png")
# 
#     fig, ax = subplots()
#     ax.boxplot([1, 2, 3], id="boxplot1")
#     scitex.io.save(fig, "/tmp/subplots_demo/boxplot1.png")
# 
#     fig, ax = subplots()
#     ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
#     scitex.io.save(fig, "/tmp/subplots_demo/bar1.png")
# 
#     print(ax.export_as_csv())
#     #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
#     # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
#     # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
#     # 2           3.0           6.0           6.0  ...                 3.0           C         6.0
# 
#     print(ax.export_as_csv().keys())  # plot3 and plot 4 are not tracked
#     # [3 rows x 11 columns]
#     # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
#     #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
#     #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
#     #       dtype='object')
# 
#     # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
#     ax.export_as_csv("./tmp/subplots_demo/for_sigmaplot.csv")
#     # Saved to: ./tmp/subplots_demo/for_sigmaplot.csv
# 
# """
# from matplotlib.pyplot import subplots as counter_part
# from scitex.plt import subplots as msubplots
# print(set(dir(msubplots)) - set(dir(counter_part)))
# is_compatible = np.all([kk in set(dir(msubplots)) for kk in set(dir(counter_part))])
# if is_compatible:
#     print(f"{msubplots.__name__} is compatible with {counter_part.__name__}")
# else:
#     print(f"{msubplots.__name__} is incompatible with {counter_part.__name__}")
# """
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_SubplotsWrapper.py
# --------------------------------------------------------------------------------
