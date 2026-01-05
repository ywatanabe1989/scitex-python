# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_stx_shaded_line.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 13:15:00 (ywatanabe)"
# # File: ./src/scitex/plt/ax/_plot/_plot_shaded_line.py
# 
# """Line plots with shaded uncertainty regions (e.g., confidence intervals)."""
# 
# from typing import Any, List, Optional, Tuple, Union
# 
# import numpy as np
# import pandas as pd
# from matplotlib.axes import Axes
# 
# from scitex.types import ColorLike
# from ....plt.utils import assert_valid_axis
# 
# 
# def _plot_single_shaded_line(
#     axis: Union[Axes, "AxisWrapper"],
#     xx: np.ndarray,
#     y_lower: np.ndarray,
#     y_middle: np.ndarray,
#     y_upper: np.ndarray,
#     color: Optional[ColorLike] = None,
#     alpha: float = 0.3,
#     **kwargs: Any,
# ) -> Tuple[Union[Axes, "AxisWrapper"], pd.DataFrame]:
#     """Plot a single line with shaded area between y_lower and y_upper bounds.
# 
#     Parameters
#     ----------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         Axes to plot on.
#     xx : np.ndarray
#         X values.
#     y_lower : np.ndarray
#         Lower bound y values.
#     y_middle : np.ndarray
#         Middle (mean/median) y values.
#     y_upper : np.ndarray
#         Upper bound y values.
#     color : ColorLike, optional
#         Color for line and fill.
#     alpha : float, default 0.3
#         Transparency for shaded region.
#     **kwargs : dict
#         Additional keyword arguments passed to plot().
# 
#     Returns
#     -------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         The axes with the plot.
#     df : pd.DataFrame
#         DataFrame with x, y_lower, y_middle, y_upper columns.
#     """
#     assert_valid_axis(
#         axis, "First argument must be a matplotlib axis or scitex axis wrapper"
#     )
#     assert len(xx) == len(y_middle) == len(y_lower) == len(y_upper), (
#         "All arrays must have the same length"
#     )
# 
#     label = kwargs.pop("label", None)
#     axis.plot(xx, y_middle, color=color, alpha=alpha, label=label, **kwargs)
#     kwargs["linewidth"] = 0
#     kwargs["edgecolor"] = "none"  # Remove edge line
#     axis.fill_between(xx, y_lower, y_upper, alpha=alpha, color=color, **kwargs)
# 
#     return axis, pd.DataFrame(
#         {"x": xx, "y_lower": y_lower, "y_middle": y_middle, "y_upper": y_upper}
#     )
# 
# 
# def _plot_shaded_line(
#     axis: Union[Axes, "AxisWrapper"],
#     xs: List[np.ndarray],
#     ys_lower: List[np.ndarray],
#     ys_middle: List[np.ndarray],
#     ys_upper: List[np.ndarray],
#     color: Optional[Union[List[ColorLike], ColorLike]] = None,
#     **kwargs: Any,
# ) -> Tuple[Union[Axes, "AxisWrapper"], List[pd.DataFrame]]:
#     """Plot multiple lines with shaded areas between ys_lower and ys_upper bounds.
# 
#     Parameters
#     ----------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         Axes to plot on.
#     xs : list of np.ndarray
#         List of x value arrays.
#     ys_lower : list of np.ndarray
#         List of lower bound y value arrays.
#     ys_middle : list of np.ndarray
#         List of middle y value arrays.
#     ys_upper : list of np.ndarray
#         List of upper bound y value arrays.
#     color : ColorLike or list of ColorLike, optional
#         Color(s) for lines and fills.
#     **kwargs : dict
#         Additional keyword arguments passed to plot().
# 
#     Returns
#     -------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         The axes with the plots.
#     results : list of pd.DataFrame
#         List of DataFrames with plot data.
#     """
#     assert_valid_axis(
#         axis, "First argument must be a matplotlib axis or scitex axis wrapper"
#     )
#     assert len(xs) == len(ys_lower) == len(ys_middle) == len(ys_upper), (
#         "All input lists must have the same length"
#     )
# 
#     results = []
#     colors = color
#     color_list = colors
# 
#     if colors is not None:
#         if not isinstance(colors, list):
#             color_list = [colors] * len(xs)
#         else:
#             assert len(colors) == len(xs), "Number of colors must match number of lines"
#             color_list = colors
# 
#         for idx, (xx, y_lower, y_middle, y_upper) in enumerate(
#             zip(xs, ys_lower, ys_middle, ys_upper)
#         ):
#             this_kwargs = kwargs.copy()
#             this_kwargs["color"] = color_list[idx]
#             _, result_df = _plot_single_shaded_line(
#                 axis, xx, y_lower, y_middle, y_upper, **this_kwargs
#             )
#             results.append(result_df)
#     else:
#         for xx, y_lower, y_middle, y_upper in zip(xs, ys_lower, ys_middle, ys_upper):
#             _, result_df = _plot_single_shaded_line(
#                 axis, xx, y_lower, y_middle, y_upper, **kwargs
#             )
#             results.append(result_df)
# 
#     return axis, results
# 
# 
# def stx_shaded_line(
#     axis: Union[Axes, "AxisWrapper"],
#     xs: Union[np.ndarray, List[np.ndarray]],
#     ys_lower: Union[np.ndarray, List[np.ndarray]],
#     ys_middle: Union[np.ndarray, List[np.ndarray]],
#     ys_upper: Union[np.ndarray, List[np.ndarray]],
#     color: Optional[Union[ColorLike, List[ColorLike]]] = None,
#     **kwargs: Any,
# ) -> Tuple[Union[Axes, "AxisWrapper"], Union[pd.DataFrame, List[pd.DataFrame]]]:
#     """Plot line(s) with shaded uncertainty regions.
# 
#     Automatically handles both single and multiple line cases. Useful for
#     plotting mean/median with confidence intervals or standard deviation bands.
# 
#     Parameters
#     ----------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         Axes to plot on.
#     xs : np.ndarray or list of np.ndarray
#         X values (single array or list of arrays for multiple lines).
#     ys_lower : np.ndarray or list of np.ndarray
#         Lower bound y values.
#     ys_middle : np.ndarray or list of np.ndarray
#         Middle (mean/median) y values.
#     ys_upper : np.ndarray or list of np.ndarray
#         Upper bound y values.
#     color : ColorLike or list of ColorLike, optional
#         Color(s) for lines and shaded regions.
#     **kwargs : dict
#         Additional keyword arguments passed to plot().
# 
#     Returns
#     -------
#     axis : matplotlib.axes.Axes or AxisWrapper
#         The axes with the plot(s).
#     data : pd.DataFrame or list of pd.DataFrame
#         DataFrame(s) containing plot data with columns:
#         x, y_lower, y_middle, y_upper.
# 
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import scitex as stx
#     >>> x = np.linspace(0, 10, 100)
#     >>> y_mean = np.sin(x)
#     >>> y_std = 0.2
#     >>> fig, ax = stx.plt.subplots()
#     >>> ax, df = stx.plt.ax.stx_shaded_line(
#     ...     ax, x, y_mean - y_std, y_mean, y_mean + y_std,
#     ...     color='blue', alpha=0.3
#     ... )
#     """
#     is_single = not (
#         isinstance(xs, list)
#         and isinstance(ys_lower, list)
#         and isinstance(ys_middle, list)
#         and isinstance(ys_upper, list)
#     )
# 
#     if is_single:
#         assert len(xs) == len(ys_lower) == len(ys_middle) == len(ys_upper), (
#             "All arrays must have the same length for single line plot"
#         )
# 
#         return _plot_single_shaded_line(
#             axis, xs, ys_lower, ys_middle, ys_upper, color=color, **kwargs
#         )
#     else:
#         return _plot_shaded_line(
#             axis, xs, ys_lower, ys_middle, ys_upper, color=color, **kwargs
#         )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_stx_shaded_line.py
# --------------------------------------------------------------------------------
