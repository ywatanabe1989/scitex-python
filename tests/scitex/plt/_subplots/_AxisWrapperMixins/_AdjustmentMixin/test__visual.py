# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_AdjustmentMixin/_visual.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: _visual.py - Visual adjustments (ticks, spines, position)
# 
# """Mixin for visual adjustments including ticks, spines, and positioning."""
# 
# import os
# from typing import List, Optional, Union
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# 
# class VisualAdjustmentMixin:
#     """Mixin for visual adjustments to axis appearance."""
# 
#     def _get_ax_module(self):
#         """Lazy import ax module to avoid circular imports."""
#         from .....plt import ax as ax_module
#         return ax_module
# 
#     def set_ticks(
#         self,
#         xvals: Optional[List[Union[int, float]]] = None,
#         xticks: Optional[List[str]] = None,
#         yvals: Optional[List[Union[int, float]]] = None,
#         yticks: Optional[List[str]] = None,
#     ) -> None:
#         """Set custom tick positions and labels.
# 
#         Parameters
#         ----------
#         xvals : list of numbers, optional
#             Positions for x-axis ticks
#         xticks : list of str, optional
#             Labels for x-axis ticks
#         yvals : list of numbers, optional
#             Positions for y-axis ticks
#         yticks : list of str, optional
#             Labels for y-axis ticks
#         """
#         self._axis_mpl = self._get_ax_module().set_ticks(
#             self._axis_mpl,
#             xvals=xvals,
#             xticks=xticks,
#             yvals=yvals,
#             yticks=yticks,
#         )
# 
#     def set_n_ticks(self, n_xticks: int = 4, n_yticks: int = 4) -> None:
#         """Set the number of ticks on each axis.
# 
#         Parameters
#         ----------
#         n_xticks : int, optional
#             Number of ticks on x-axis, by default 4
#         n_yticks : int, optional
#             Number of ticks on y-axis, by default 4
#         """
#         self._axis_mpl = self._get_ax_module().set_n_ticks(
#             self._axis_mpl, n_xticks=n_xticks, n_yticks=n_yticks
#         )
# 
#     def hide_spines(
#         self,
#         top: bool = True,
#         bottom: bool = False,
#         left: bool = False,
#         right: bool = True,
#         ticks: bool = False,
#         labels: bool = False,
#     ) -> None:
#         """Hide specific spines and optionally ticks/labels.
# 
#         Parameters
#         ----------
#         top : bool, optional
#             Hide top spine, by default True
#         bottom : bool, optional
#             Hide bottom spine, by default False
#         left : bool, optional
#             Hide left spine, by default False
#         right : bool, optional
#             Hide right spine, by default True
#         ticks : bool, optional
#             Hide all ticks, by default False
#         labels : bool, optional
#             Hide all tick labels, by default False
#         """
#         self._axis_mpl = self._get_ax_module().hide_spines(
#             self._axis_mpl,
#             top=top,
#             bottom=bottom,
#             left=left,
#             right=right,
#             ticks=ticks,
#             labels=labels,
#         )
# 
#     def extend(self, x_ratio: float = 1.0, y_ratio: float = 1.0) -> None:
#         """Extend axis limits by a ratio.
# 
#         Parameters
#         ----------
#         x_ratio : float, optional
#             Ratio to extend x-axis by, by default 1.0
#         y_ratio : float, optional
#             Ratio to extend y-axis by, by default 1.0
#         """
#         self._axis_mpl = self._get_ax_module().extend(
#             self._axis_mpl, x_ratio=x_ratio, y_ratio=y_ratio
#         )
# 
#     def shift(self, dx: float = 0, dy: float = 0) -> None:
#         """Shift axis position.
# 
#         Parameters
#         ----------
#         dx : float, optional
#             Horizontal shift, by default 0
#         dy : float, optional
#             Vertical shift, by default 0
#         """
#         self._axis_mpl = self._get_ax_module().shift(self._axis_mpl, dx=dx, dy=dy)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_AdjustmentMixin/_visual.py
# --------------------------------------------------------------------------------
