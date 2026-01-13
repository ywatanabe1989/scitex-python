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
# """SubplotsWrapper: Monitor data plotted using matplotlib for CSV export."""
# 
# import os
# from collections import OrderedDict
# 
# import matplotlib.pyplot as plt
# 
# __FILE__ = "./src/scitex/plt/_subplots/_SubplotsWrapper.py"
# __DIR__ = os.path.dirname(__FILE__)
# 
# # Configure fonts at import
# from ._fonts import _arial_enabled  # noqa: F401
# from ._mm_layout import create_with_mm_control
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
# 
#     Supports optional figrecipe integration for reproducible figures.
#     When figrecipe is available and `use_figrecipe=True`, figures are created
#     with recipe recording capability for later reproduction.
#     """
# 
#     def __init__(self):
#         self._subplots_wrapper_history = OrderedDict()
#         self._fig_scitex = None
#         self._counter_part = plt.subplots
#         self._figrecipe_available = None  # Lazy check
# 
#     def _check_figrecipe(self):
#         """Check if figrecipe is available (lazy, cached)."""
#         if self._figrecipe_available is None:
#             try:
#                 import figrecipe  # noqa: F401
# 
#                 self._figrecipe_available = True
#             except ImportError:
#                 self._figrecipe_available = False
#         return self._figrecipe_available
# 
#     def __call__(
#         self,
#         *args,
#         track=True,
#         sharex=False,
#         sharey=False,
#         constrained_layout=None,
#         use_figrecipe=None,  # NEW: Enable figrecipe recording
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
#         styles=None,
#         transparent=None,
#         theme=None,
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
#         use_figrecipe : bool or None, optional
#             If True, use figrecipe for recipe recording.
#             If None (default), auto-detect figrecipe availability.
#             If False, disable figrecipe even if available.
# 
#         MM-Control Parameters
#         ---------------------
#         axes_width_mm, axes_height_mm : float or list
#             Axes dimensions in mm
#         margin_*_mm : float
#             Figure margins in mm
#         space_w_mm, space_h_mm : float
#             Spacing between axes in mm
#         mode : str
#             'publication' or 'display'
# 
#         Returns
#         -------
#         fig : FigWrapper
#             Wrapped matplotlib Figure (with optional RecordingFigure)
#         ax or axes : AxisWrapper or AxesWrapper
#             Wrapped matplotlib Axes
#         """
#         # Resolve style values
#         from scitex.plt.styles import SCITEX_STYLE as _S
#         from scitex.plt.styles import resolve_style_value as _resolve
# 
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
# 
#         if transparent is None:
#             transparent = _S.get("transparent", True)
#         if mode is None:
#             mode = _S.get("mode", "publication")
#         if theme is None:
#             theme = _resolve("theme.mode", None, "light", str)
# 
#         # Determine figrecipe usage
#         if use_figrecipe is None:
#             use_figrecipe = self._check_figrecipe()
# 
#         # Create figure with mm-control
#         fig, axes = create_with_mm_control(
#             *args,
#             track=track,
#             sharex=sharex,
#             sharey=sharey,
#             axes_width_mm=axes_width_mm,
#             axes_height_mm=axes_height_mm,
#             margin_left_mm=margin_left_mm,
#             margin_right_mm=margin_right_mm,
#             margin_bottom_mm=margin_bottom_mm,
#             margin_top_mm=margin_top_mm,
#             space_w_mm=space_w_mm,
#             space_h_mm=space_h_mm,
#             axes_thickness_mm=axes_thickness_mm,
#             tick_length_mm=tick_length_mm,
#             tick_thickness_mm=tick_thickness_mm,
#             trace_thickness_mm=trace_thickness_mm,
#             marker_size_mm=marker_size_mm,
#             axis_font_size_pt=axis_font_size_pt,
#             tick_font_size_pt=tick_font_size_pt,
#             title_font_size_pt=title_font_size_pt,
#             legend_font_size_pt=legend_font_size_pt,
#             suptitle_font_size_pt=suptitle_font_size_pt,
#             n_ticks=n_ticks,
#             mode=mode,
#             dpi=dpi,
#             styles=styles,
#             transparent=transparent,
#             theme=theme,
#             **kwargs,
#         )
# 
#         # If figrecipe enabled, create recording layer
#         if use_figrecipe:
#             self._attach_figrecipe_recorder(fig)
# 
#         self._fig_scitex = fig
#         return fig, axes
# 
#     def _attach_figrecipe_recorder(self, fig_wrapper):
#         """Attach figrecipe recorder to FigWrapper for recipe export.
# 
#         This creates a RecordingFigure layer that wraps the underlying
#         matplotlib figure, enabling save_recipe() on the FigWrapper.
#         """
#         try:
#             from figrecipe._recorder import Recorder
# 
#             # Get the underlying matplotlib figure
#             mpl_fig = fig_wrapper._fig_mpl
# 
#             # Create recorder
#             recorder = Recorder()
#             figsize = mpl_fig.get_size_inches()
#             dpi_val = mpl_fig.dpi
#             recorder.start_figure(figsize=tuple(figsize), dpi=int(dpi_val))
# 
#             # Store recorder on FigWrapper for later recipe export
#             fig_wrapper._figrecipe_recorder = recorder
#             fig_wrapper._figrecipe_enabled = True
# 
#             # Store style info from scitex in the recipe
#             if hasattr(mpl_fig, "_scitex_theme"):
#                 recorder.figure_record.style = {"theme": mpl_fig._scitex_theme}
# 
#         except Exception:
#             # Silently fail - figrecipe is optional
#             fig_wrapper._figrecipe_enabled = False
# 
#     def __dir__(self):
#         """Provide combined directory for tab completion."""
#         local_attrs = set(super().__dir__())
#         try:
#             counterpart_attrs = set(dir(self._counter_part))
#         except Exception:
#             counterpart_attrs = set()
#         return sorted(local_attrs.union(counterpart_attrs))
# 
# 
# # Instantiate the wrapper
# subplots = SubplotsWrapper()
# 
# 
# if __name__ == "__main__":
#     import matplotlib
# 
#     import scitex
# 
#     matplotlib.use("TkAgg")
# 
#     fig, ax = subplots()
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
#     scitex.io.save(fig, "/tmp/subplots_demo/plots.png")
# 
#     print(ax.export_as_csv())
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_SubplotsWrapper.py
# --------------------------------------------------------------------------------
