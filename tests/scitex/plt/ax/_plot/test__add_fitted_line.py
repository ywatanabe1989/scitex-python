# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_add_fitted_line.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-19 15:52:00 (ywatanabe)"
# # File: ./src/scitex/plt/ax/_plot/_add_fitted_line.py
# 
# """
# Add fitted regression line to scatter plots.
# """
# 
# import numpy as np
# from typing import Optional, Tuple, Dict
# 
# 
# def add_fitted_line(
#     ax,
#     x,
#     y,
#     color: str = "black",
#     linestyle: str = "--",
#     linewidth_mm: float = 0.2,
#     label: Optional[str] = None,
#     degree: int = 1,
#     show_stats: bool = True,
#     stats_position: float = 0.75,
#     stats_fontsize: int = 6,
# ) -> Tuple:
#     """
#     Add a fitted polynomial line to a scatter plot with optional R² and p-value.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         Axes to plot on
#     x : array-like
#         X data
#     y : array-like
#         Y data
#     color : str, optional
#         Line color (default: 'black')
#     linestyle : str, optional
#         Line style (default: '--' for dashed)
#     linewidth_mm : float, optional
#         Line thickness in millimeters (default: 0.2mm)
#     label : str, optional
#         Label for the fitted line (default: None)
#     degree : int, optional
#         Polynomial degree for fitting (default: 1 for linear)
#     show_stats : bool, optional
#         Whether to display R² and p-value near the line (default: True)
#         Only applicable for linear fits (degree=1)
#     stats_position : float, optional
#         Position along x-axis (0-1 scale) for stats text (default: 0.75)
#     stats_fontsize : int, optional
#         Font size for statistics text in points (default: 6)
# 
#     Returns
#     -------
#     line : Line2D
#         The fitted line object
#     coeffs : np.ndarray
#         Polynomial coefficients from np.polyfit
#     stats : StatResult or None
#         StatResult instance with correlation statistics (only for degree=1).
#         Use .to_dict() for dictionary format.
# 
#     Examples
#     --------
#     >>> fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
#     >>> scatter = ax.scatter(x, y)
#     >>> stx.plt.ax.add_fitted_line(ax, x, y)  # Auto-shows R² and p
# 
#     >>> # Without statistics
#     >>> line, coeffs, stats = stx.plt.ax.add_fitted_line(
#     ...     ax, x, y, show_stats=False
#     ... )
# 
#     >>> # Custom position for stats
#     >>> line, coeffs, stats = stx.plt.ax.add_fitted_line(
#     ...     ax, x, y, stats_position=0.5
#     ... )
#     """
#     from scitex.plt.utils import mm_to_pt
# 
#     # Convert data to numpy arrays
#     x = np.asarray(x)
#     y = np.asarray(y)
# 
#     # Fit polynomial
#     coeffs = np.polyfit(x, y, degree)
#     poly_fn = np.poly1d(coeffs)
# 
#     # Generate fitted line points
#     x_fit = np.linspace(x.min(), x.max(), 100)
#     y_fit = poly_fn(x_fit)
# 
#     # Convert linewidth to points
#     lw_pt = mm_to_pt(linewidth_mm)
# 
#     # Plot fitted line
#     line = ax.plot(
#         x_fit,
#         y_fit,
#         color=color,
#         linestyle=linestyle,
#         linewidth=lw_pt,
#         label=label,
#     )[0]
# 
#     # Calculate and display statistics for linear regression (degree=1)
#     stats_result = None
#     if degree == 1 and show_stats:
#         # Import scitex.stats correlation test
#         from scitex.stats.tests.correlation import test_pearson
# 
#         # Calculate correlation statistics using scitex.stats
#         stats_result = test_pearson(x, y)
# 
#         # Position for text annotation
#         x_pos = x.min() + stats_position * (x.max() - x.min())
#         y_pos = poly_fn(x_pos)
# 
#         # Format statistics text with R² and significance stars
#         r_squared = stats_result.effect_size["value"]  # r_squared from effect_size
#         stars = stats_result.stars
# 
#         if stars and stars != "ns":  # Only show if significant
#             stats_text = f"$R^2$ = {r_squared:.3f}{stars}"
#         else:  # Not significant
#             stats_text = f"$R^2$ = {r_squared:.3f} (ns)"
# 
#         # Add text annotation near the line
#         ax.text(
#             x_pos,
#             y_pos,
#             stats_text,
#             verticalalignment="bottom",
#             fontsize=stats_fontsize,
#         )
# 
#         # Store stats in axes metadata for embedding in saved figures
#         if not hasattr(ax, "_scitex_metadata"):
#             ax._scitex_metadata = {}
#         if "stats" not in ax._scitex_metadata:
#             ax._scitex_metadata["stats"] = []
# 
#         # Add this StatResult to the stats list
#         ax._scitex_metadata["stats"].append(stats_result.to_dict())
# 
#     return line, coeffs, stats_result
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_add_fitted_line.py
# --------------------------------------------------------------------------------
