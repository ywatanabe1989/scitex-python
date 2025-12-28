# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_style/_auto_scale_axis.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2025-11-19 18:45:00 (ywatanabe)"
# # File: ./src/scitex/plt/ax/_style/_auto_scale_axis.py
# 
# """
# Automatic axis scaling to factor out common powers of 10.
# 
# This utility automatically detects when axis tick values are very small or very large
# and factors out the appropriate power of 10, updating both the tick labels and axis label.
# 
# Examples:
#     0.0000, 0.0008, 0.0016, 0.0024  →  0, 0.8, 1.6, 2.4  with label "[×10⁻³]"
#     10000, 20000, 30000, 40000      →  10, 20, 30, 40    with label "[×10³]"
# """
# 
# import numpy as np
# from typing import Optional, Tuple
# 
# 
# def detect_scale_factor(
#     values: np.ndarray, threshold: float = 1e-2
# ) -> Tuple[int, bool]:
#     """
#     Detect appropriate power of 10 to factor out from axis values.
# 
#     Parameters
#     ----------
#     values : np.ndarray
#         Array of tick values on the axis
#     threshold : float
#         Threshold below which we consider factoring out (default: 0.01)
# 
#     Returns
#     -------
#     power : int
#         Power of 10 to factor out (e.g., -3 for values like 0.001-0.009)
#     should_scale : bool
#         Whether scaling should be applied
# 
#     Examples
#     --------
#     >>> detect_scale_factor(np.array([0.0, 0.0008, 0.0016, 0.0024]))
#     (-3, True)
#     >>> detect_scale_factor(np.array([10000, 20000, 30000]))
#     (3, True)
#     >>> detect_scale_factor(np.array([0, 1, 2, 3]))
#     (0, False)
#     """
#     # Filter out zero values for calculation
#     nonzero_values = values[values != 0]
# 
#     if len(nonzero_values) == 0:
#         return 0, False
# 
#     # Get the order of magnitude of the maximum absolute value
#     max_abs = np.max(np.abs(nonzero_values))
# 
#     # Check if values are very small (< threshold) or very large (> 1/threshold)
#     if max_abs < threshold:
#         # Values are very small - factor out negative power
#         power = int(np.floor(np.log10(max_abs)))
#         return power, True
#     elif max_abs > 1.0 / threshold:
#         # Values are very large - factor out positive power
#         power = int(np.floor(np.log10(max_abs)))
#         # Only scale if power >= 3 (thousands or larger)
#         if power >= 3:
#             return power, True
# 
#     return 0, False
# 
# 
# def format_scale_factor(power: int) -> str:
#     """
#     Format the scale factor for display in axis label.
# 
#     Parameters
#     ----------
#     power : int
#         Power of 10 (e.g., -3, 3, 6)
# 
#     Returns
#     -------
#     str
#         Formatted string using matplotlib mathtext (e.g., "×10$^{-3}$", "×10$^{6}$")
# 
#     Examples
#     --------
#     >>> format_scale_factor(-3)
#     '×10$^{-3}$'
#     >>> format_scale_factor(6)
#     '×10$^{6}$'
#     """
#     if power == 0:
#         return ""
# 
#     # Use matplotlib's mathtext for reliable rendering across all formats
#     return f"×10$^{{{power}}}$"
# 
# 
# def auto_scale_axis(ax, axis: str = "both", threshold: float = 1e-2) -> None:
#     """
#     Automatically scale axis to factor out common powers of 10.
# 
#     This function:
#     1. Detects when tick values are very small or very large
#     2. Factors out the appropriate power of 10
#     3. Updates tick labels to show factored values
#     4. Appends the scale factor to the axis label
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         Axes object to apply scaling to
#     axis : str, optional
#         Which axis to scale: 'x', 'y', or 'both' (default: 'both')
#     threshold : float, optional
#         Threshold for triggering scaling (default: 1e-2)
#         Values with max < threshold or max > 1/threshold will be scaled
# 
#     Examples
#     --------
#     >>> import matplotlib.pyplot as plt
#     >>> fig, ax = plt.subplots()
#     >>> ax.plot([0, 1, 2], [0.0001, 0.0002, 0.0003])
#     >>> ax.set_ylabel('Density')
#     >>> auto_scale_axis(ax, axis='y')
#     >>> # Y-axis now shows: 0.1, 0.2, 0.3 with label "Density [×10⁻³]"
# 
#     Notes
#     -----
#     - Only scales if the range of values justifies it (very small or very large)
#     - Preserves the original axis label and appends the scale factor
#     - Uses Unicode superscripts for clean display (×10⁻³, ×10⁶, etc.)
#     """
#     import matplotlib.ticker as ticker
# 
#     def scale_axis_impl(ax_obj, is_x_axis: bool):
#         """Internal implementation for scaling a single axis."""
#         # Get current tick values
#         if is_x_axis:
#             tick_values = np.array(ax_obj.get_xticks())
#             get_label = ax_obj.get_xlabel
#             set_label = ax_obj.set_xlabel
#             set_formatter = ax_obj.xaxis.set_major_formatter
#         else:
#             tick_values = np.array(ax_obj.get_yticks())
#             get_label = ax_obj.get_ylabel
#             set_label = ax_obj.set_ylabel
#             set_formatter = ax_obj.yaxis.set_major_formatter
# 
#         # Detect if scaling is needed
#         power, should_scale = detect_scale_factor(tick_values, threshold)
# 
#         if not should_scale:
#             return
# 
#         # Create scaling factor
#         scale_factor = 10**power
# 
#         # Update tick formatter to show scaled values
#         def format_func(value, pos):
#             scaled_value = value / scale_factor
#             # Format with appropriate precision
#             if abs(scaled_value) < 10:
#                 return f"{scaled_value:.1f}"
#             else:
#                 return f"{scaled_value:.0f}"
# 
#         set_formatter(ticker.FuncFormatter(format_func))
# 
#         # Update axis label with scale factor
#         current_label = get_label()
#         scale_str = format_scale_factor(power)
# 
#         # Check if label already has units in brackets
#         if "[" in current_label and "]" in current_label:
#             # Insert scale factor before the closing bracket
#             # e.g., "Density [a.u.]" → "Density [×10⁻³ a.u.]"
#             label_parts = current_label.rsplit("]", 1)
#             new_label = f"{label_parts[0]} {scale_str}]{label_parts[1]}"
#         else:
#             # Append scale factor in brackets
#             # e.g., "Density" → "Density [×10⁻³]"
#             new_label = (
#                 f"{current_label} [{scale_str}]" if current_label else f"[{scale_str}]"
#             )
# 
#         set_label(new_label)
# 
#     # Apply to requested axes
#     if axis in ["x", "both"]:
#         scale_axis_impl(ax, is_x_axis=True)
#     if axis in ["y", "both"]:
#         scale_axis_impl(ax, is_x_axis=False)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_style/_auto_scale_axis.py
# --------------------------------------------------------------------------------
