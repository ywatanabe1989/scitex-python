#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 09:02:33 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/tests/scitex/plt/ax/_adjust/test__rotate_labels.py
# ----------------------------------------
import os

__FILE__ = "./tests/scitex/plt/ax/_adjust/test__rotate_labels.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
pytest.importorskip("zarr")

matplotlib.use("Agg")

from scitex.plt.ax._style import rotate_labels


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Create a basic plot with labels
        xx = np.linspace(0, 10, 5)
        yy = np.sin(xx)
        self.ax.plot(xx, yy)
        self.ax.set_xticks(xx)
        self.ax.set_yticks(yy)

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test with rotation parameters (default args don't rotate)
        ax = rotate_labels(self.ax, x=45, y=45)

        # Force draw to ensure labels are updated
        self.fig.canvas.draw()

        # Check x and y tick label rotations
        for label in ax.get_xticklabels():
            assert label.get_rotation() == 45

        for label in ax.get_yticklabels():
            assert label.get_rotation() == 45

    def test_custom_rotations(self):
        # Test with custom rotation angles
        ax = rotate_labels(self.ax, x=30, y=60)

        # Force draw to ensure labels are updated
        self.fig.canvas.draw()

        # Check custom x and y tick label rotations
        for label in ax.get_xticklabels():
            assert label.get_rotation() == 30

        for label in ax.get_yticklabels():
            assert label.get_rotation() == 60

    def test_custom_alignment(self):
        # Test with custom horizontal alignments (must also provide rotation)
        ax = rotate_labels(self.ax, x=45, y=45, x_ha="left", y_ha="right")

        # Force draw to ensure labels are updated
        self.fig.canvas.draw()

        # Check custom alignments
        for label in ax.get_xticklabels():
            assert label.get_ha() == "left"

        for label in ax.get_yticklabels():
            assert label.get_ha() == "right"

    def test_rotate_x_only(self):
        # Test rotating only x labels
        ax = rotate_labels(self.ax, x=90, y=0)

        # Force draw to ensure labels are updated
        self.fig.canvas.draw()

        # Check that x labels are rotated but y labels are vertical
        for label in ax.get_xticklabels():
            assert label.get_rotation() == 90

        for label in ax.get_yticklabels():
            assert label.get_rotation() == 0

    def test_rotate_y_only(self):
        # Test rotating only y labels
        ax = rotate_labels(self.ax, x=0, y=90)

        # Force draw to ensure labels are updated
        self.fig.canvas.draw()

        # Check that y labels are rotated but x labels are horizontal
        for label in ax.get_xticklabels():
            assert label.get_rotation() == 0

        for label in ax.get_yticklabels():
            assert label.get_rotation() == 90

    def test_savefig(self):
        from scitex.io import save

        # Main test functionality
        rotate_labels(self.ax, x=45, y=30)

        # Saving
        spath = f"./{os.path.basename(__file__)}.jpg"
        save(self.fig, spath)

        # Check saved file
        ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
        actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
        assert os.path.exists(actual_spath), f"Failed to save figure to {spath}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_style/_rotate_labels.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-24 13:22:52 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/plt/ax/_style/_rotate_labels.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# # Time-stamp: "2024-10-27 13:24:32 (ywatanabe)"
# # /home/ywatanabe/proj/_scitex_repo_openhands/src/scitex/plt/ax/_rotate_labels.py
# 
# """This script does XYZ."""
# 
# """Imports"""
# import numpy as np
# 
# 
# def rotate_labels(
#     ax,
#     x=None,
#     y=None,
#     x_ha=None,
#     y_ha=None,
#     x_va=None,
#     y_va=None,
#     auto_adjust=True,
#     scientific_convention=True,
#     tight_layout=False,
# ):
#     """
#     Rotate x and y axis labels of a matplotlib Axes object with automatic positioning.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The Axes object to modify.
#     x : float or None, optional
#         Rotation angle for x-axis labels in degrees. Default is None.
#         If 0 or None, x-axis labels are not rotated.
#     y : float or None, optional
#         Rotation angle for y-axis labels in degrees. Default is None.
#         If 0 or None, y-axis labels are not rotated.
#     x_ha : str, optional
#         Horizontal alignment for x-axis labels. If None, automatically determined.
#     y_ha : str, optional
#         Horizontal alignment for y-axis labels. If None, automatically determined.
#     x_va : str, optional
#         Vertical alignment for x-axis labels. If None, automatically determined.
#     y_va : str, optional
#         Vertical alignment for y-axis labels. If None, automatically determined.
#     auto_adjust : bool, optional
#         Whether to automatically adjust alignment based on rotation angle. Default is True.
#     scientific_convention : bool, optional
#         Whether to follow scientific plotting conventions. Default is True.
#     tight_layout : bool, optional
#         Whether to apply tight_layout to prevent overlapping. Default is False.
# 
#     Returns
#     -------
#     matplotlib.axes.Axes
#         The modified Axes object.
# 
#     Example
#     -------
#     fig, ax = plt.subplots()
#     ax.plot([1, 2, 3], [1, 2, 3])
#     rotate_labels(ax)
#     plt.show()
# 
#     Notes
#     -----
#     Scientific conventions for label rotation:
#     - X-axis labels: For angles 0-90°, use 'right' alignment; for 90-180°, use 'left'
#     - Y-axis labels: For angles 0-90°, use 'center' alignment; adjust vertical as needed
#     - Optimal readability maintained through automatic positioning
#     """
#     # Determine which axes to rotate (skip if None or 0)
#     rotate_x = x is not None and x != 0
#     rotate_y = y is not None and y != 0
# 
#     # Get current tick positions
#     xticks = ax.get_xticks()
#     yticks = ax.get_yticks()
# 
#     # Set ticks explicitly
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)
# 
#     # Auto-adjust alignment based on rotation angle and scientific conventions
#     if auto_adjust:
#         if rotate_x:
#             x_ha, x_va = _get_optimal_alignment(
#                 "x", x, x_ha, x_va, scientific_convention
#             )
#         if rotate_y:
#             y_ha, y_va = _get_optimal_alignment(
#                 "y", y, y_ha, y_va, scientific_convention
#             )
# 
#     # Apply defaults if not auto-adjusting
#     if rotate_x:
#         if x_ha is None:
#             x_ha = "center"
#         if x_va is None:
#             x_va = "center"
#     if rotate_y:
#         if y_ha is None:
#             y_ha = "center"
#         if y_va is None:
#             y_va = "center"
# 
#     # Check if this axis is part of a shared x-axis configuration
#     # If labels are already visible (bottom subplot or not shared), keep them visible
#     # This preserves matplotlib's default sharex behavior
#     x_labels_visible = ax.xaxis.get_tick_params()["labelbottom"]
#     y_labels_visible = ax.yaxis.get_tick_params()["labelleft"]
# 
#     # Set labels with rotation and proper alignment
#     # Only set labels if they're currently visible (respects sharex/sharey)
#     if x_labels_visible and rotate_x:
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=x, ha=x_ha, va=x_va)
#     if y_labels_visible and rotate_y:
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=y, ha=y_ha, va=y_va)
# 
#     # Auto-adjust subplot parameters for better layout if needed
#     if auto_adjust and scientific_convention:
#         # Only pass non-zero angles for adjustment
#         x_angle = x if rotate_x else 0
#         y_angle = y if rotate_y else 0
#         _adjust_subplot_params(ax, x_angle, y_angle)
# 
#     # Apply tight_layout if requested to prevent overlapping
#     if tight_layout:
#         fig = ax.get_figure()
#         try:
#             fig.tight_layout()
#         except Exception:
#             # Fallback to manual adjustment if tight_layout fails
#             x_angle = x if rotate_x else 0
#             y_angle = y if rotate_y else 0
#             _adjust_subplot_params(ax, x_angle, y_angle)
# 
#     return ax
# 
# 
# def _get_optimal_alignment(axis, angle, ha, va, scientific_convention):
#     """
#     Determine optimal alignment based on rotation angle and scientific conventions.
# 
#     Parameters
#     ----------
#     axis : str
#         'x' or 'y' axis
#     angle : float
#         Rotation angle in degrees
#     ha : str or None
#         Current horizontal alignment
#     va : str or None
#         Current vertical alignment
#     scientific_convention : bool
#         Whether to follow scientific conventions
# 
#     Returns
#     -------
#     tuple
#         (horizontal_alignment, vertical_alignment)
#     """
#     # Normalize angle to 0-360 range
#     angle = angle % 360
# 
#     if axis == "x":
#         if scientific_convention:
#             # Scientific convention for x-axis labels
#             if 0 <= angle <= 30:
#                 ha = ha or "center"
#                 va = va or "top"
#             elif 30 < angle <= 60:
#                 ha = ha or "right"
#                 va = va or "top"
#             elif 60 < angle < 90:
#                 ha = ha or "right"
#                 va = va or "top"
#             elif angle == 90:
#                 # Special case for exact 90 degrees
#                 ha = ha or "right"
#                 va = va or "top"
#             elif 90 < angle <= 120:
#                 ha = ha or "right"
#                 va = va or "center"
#             elif 120 < angle <= 150:
#                 ha = ha or "right"
#                 va = va or "bottom"
#             elif 150 < angle <= 210:
#                 ha = ha or "center"
#                 va = va or "bottom"
#             elif 210 < angle <= 240:
#                 ha = ha or "left"
#                 va = va or "bottom"
#             elif 240 < angle <= 300:
#                 ha = ha or "left"
#                 va = va or "center"
#             else:  # 300-360
#                 ha = ha or "left"
#                 va = va or "top"
#         else:
#             ha = ha or "center"
#             va = va or "top"
# 
#     else:  # y-axis
#         if scientific_convention:
#             # Scientific convention for y-axis labels
#             if 0 <= angle <= 30:
#                 ha = ha or "right"
#                 va = va or "center"
#             elif 30 < angle <= 60:
#                 ha = ha or "right"
#                 va = va or "bottom"
#             elif 60 < angle <= 120:
#                 ha = ha or "center"
#                 va = va or "bottom"
#             elif 120 < angle <= 150:
#                 ha = ha or "left"
#                 va = va or "bottom"
#             elif 150 < angle <= 210:
#                 ha = ha or "left"
#                 va = va or "center"
#             elif 210 < angle <= 240:
#                 ha = ha or "left"
#                 va = va or "top"
#             elif 240 < angle <= 300:
#                 ha = ha or "center"
#                 va = va or "top"
#             else:  # 300-360
#                 ha = ha or "right"
#                 va = va or "top"
#         else:
#             ha = ha or "center"
#             va = va or "center"
# 
#     return ha, va
# 
# 
# def _adjust_subplot_params(ax, x_angle, y_angle):
#     """
#     Automatically adjust subplot parameters to accommodate rotated labels.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes object
#     x_angle : float
#         X-axis rotation angle
#     y_angle : float
#         Y-axis rotation angle
#     """
#     fig = ax.get_figure()
# 
#     # Check if figure is using a layout engine that is incompatible with subplots_adjust
#     try:
#         # For matplotlib >= 3.6
#         if hasattr(fig, "get_layout_engine"):
#             layout_engine = fig.get_layout_engine()
#             if layout_engine is not None:
#                 # If using constrained_layout or tight_layout, don't adjust
#                 return
#     except AttributeError:
#         pass
# 
#     # Check for constrained_layout (older matplotlib versions)
#     try:
#         if hasattr(fig, "get_constrained_layout"):
#             if fig.get_constrained_layout():
#                 # Constrained layout is active, don't adjust
#                 return
#     except AttributeError:
#         pass
# 
#     # Calculate required margins based on rotation angles
#     # Special handling for 90-degree rotation
#     if x_angle == 90:
#         x_margin_factor = 0.3  # Maximum margin for 90 degrees
#     else:
#         # Increase margin more significantly for rotated x-axis labels to prevent xlabel overlap
#         x_margin_factor = abs(np.sin(np.radians(x_angle))) * 0.25  # Increased from 0.2
# 
#     y_margin_factor = abs(np.sin(np.radians(y_angle))) * 0.15
# 
#     # Get current subplot parameters
#     try:
#         subplotpars = fig.subplotpars
#         current_bottom = subplotpars.bottom
#         current_left = subplotpars.left
# 
#         # Adjust margins if they need to be increased
#         # Ensure more space for rotated x-labels and xlabel
#         new_bottom = max(
#             current_bottom, 0.2 + x_margin_factor
#         )  # Increased base from 0.15
#         new_left = max(current_left, 0.1 + y_margin_factor)
# 
#         # Only adjust if we're increasing the margins significantly
#         if (
#             new_bottom > current_bottom + 0.02 or new_left > current_left + 0.02
#         ):  # Reduced threshold
#             # Suppress warning and try to adjust
#             import warnings
# 
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 fig.subplots_adjust(bottom=new_bottom, left=new_left)
#     except Exception:
#         # Skip adjustment if there are issues
#         pass
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_style/_rotate_labels.py
# --------------------------------------------------------------------------------
