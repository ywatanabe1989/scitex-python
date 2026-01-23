# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/_handlers_plot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-13
# # File: src/scitex/plt/_mcp/_handlers_plot.py
# 
# """Plot-related MCP handlers for SciTeX plt module."""
# 
# from __future__ import annotations
# 
# from typing import Optional
# 
# from ._handlers_figure import _get_axes
# 
# 
# async def plot_bar_handler(
#     x: list[str],
#     y: list[float],
#     figure_id: Optional[str] = None,
#     panel: str = "0,0",
#     yerr: Optional[list[float]] = None,
#     colors: Optional[list[str]] = None,
#     xlabel: Optional[str] = None,
#     ylabel: Optional[str] = None,
#     title: Optional[str] = None,
# ) -> dict:
#     """Create a bar plot on specified panel."""
#     try:
#         fig, ax, fid = _get_axes(figure_id, panel)
# 
#         if colors is None:
#             from scitex.plt import color as color_module
# 
#             params = getattr(color_module, "PARAMS", {})
#             rgba_cycle = params.get("RGBA_NORM_FOR_CYCLE", {})
#             color_list = list(rgba_cycle.values())
#             colors = color_list[: len(x)] if color_list else None
# 
#         ax.bar(x, y, yerr=yerr, capsize=3, color=colors)
# 
#         if xlabel:
#             ax.set_xlabel(xlabel)
#         if ylabel:
#             ax.set_ylabel(ylabel)
#         if title:
#             ax.set_title(title)
# 
#         return {
#             "success": True,
#             "figure_id": fid,
#             "panel": panel,
#             "plot_type": "bar",
#             "n_bars": len(x),
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def plot_scatter_handler(
#     x: list[float],
#     y: list[float],
#     figure_id: Optional[str] = None,
#     panel: str = "0,0",
#     color: Optional[str] = None,
#     size: Optional[float] = None,
#     alpha: float = 0.7,
#     add_regression: bool = False,
#     xlabel: Optional[str] = None,
#     ylabel: Optional[str] = None,
#     title: Optional[str] = None,
# ) -> dict:
#     """Create a scatter plot on specified panel."""
#     try:
#         import numpy as np
# 
#         fig, ax, fid = _get_axes(figure_id, panel)
# 
#         if color is None:
#             color = "#c633ff"  # SciTeX purple
# 
#         s = (size * 2.83465) ** 2 if size else 15
# 
#         ax.scatter(x, y, c=color, s=s, alpha=alpha)
# 
#         if add_regression:
#             z = np.polyfit(x, y, 1)
#             p = np.poly1d(z)
#             x_line = np.linspace(min(x), max(x), 100)
#             ax.plot(x_line, p(x_line), color="#e25e33", linestyle="--", linewidth=1)
# 
#         if xlabel:
#             ax.set_xlabel(xlabel)
#         if ylabel:
#             ax.set_ylabel(ylabel)
#         if title:
#             ax.set_title(title)
# 
#         return {
#             "success": True,
#             "figure_id": fid,
#             "panel": panel,
#             "plot_type": "scatter",
#             "n_points": len(x),
#             "regression_added": add_regression,
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def plot_line_handler(
#     x: list[float],
#     y: list[float],
#     figure_id: Optional[str] = None,
#     panel: str = "0,0",
#     yerr: Optional[list[float]] = None,
#     color: Optional[str] = None,
#     label: Optional[str] = None,
#     linestyle: str = "-",
#     xlabel: Optional[str] = None,
#     ylabel: Optional[str] = None,
#     title: Optional[str] = None,
# ) -> dict:
#     """Create a line plot on specified panel."""
#     try:
#         fig, ax, fid = _get_axes(figure_id, panel)
# 
#         if color is None:
#             color = "#007fbf"  # SciTeX blue
# 
#         ax.plot(x, y, color=color, label=label, linestyle=linestyle)
# 
#         if yerr:
#             import numpy as np
# 
#             y_arr = np.array(y)
#             yerr_arr = np.array(yerr)
#             ax.fill_between(
#                 x, y_arr - yerr_arr, y_arr + yerr_arr, alpha=0.3, color=color
#             )
# 
#         if xlabel:
#             ax.set_xlabel(xlabel)
#         if ylabel:
#             ax.set_ylabel(ylabel)
#         if title:
#             ax.set_title(title)
#         if label:
#             ax.legend(loc="upper right", frameon=False)
# 
#         return {
#             "success": True,
#             "figure_id": fid,
#             "panel": panel,
#             "plot_type": "line",
#             "n_points": len(x),
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def plot_box_handler(
#     data: list[list[float]],
#     figure_id: Optional[str] = None,
#     panel: str = "0,0",
#     labels: Optional[list[str]] = None,
#     colors: Optional[list[str]] = None,
#     xlabel: Optional[str] = None,
#     ylabel: Optional[str] = None,
#     title: Optional[str] = None,
# ) -> dict:
#     """Create a box plot on specified panel."""
#     try:
#         fig, ax, fid = _get_axes(figure_id, panel)
# 
#         bp = ax.boxplot(data, patch_artist=True, widths=0.6)
# 
#         if colors is None:
#             colors = ["#007fbf", "#ff4433", "#14b514", "#c633ff", "#e25e33"]
#         for i, box in enumerate(bp["boxes"]):
#             box.set_facecolor(colors[i % len(colors)])
# 
#         if labels:
#             ax.set_xticklabels(labels)
#         if xlabel:
#             ax.set_xlabel(xlabel)
#         if ylabel:
#             ax.set_ylabel(ylabel)
#         if title:
#             ax.set_title(title)
# 
#         return {
#             "success": True,
#             "figure_id": fid,
#             "panel": panel,
#             "plot_type": "box",
#             "n_groups": len(data),
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def plot_violin_handler(
#     data: list[list[float]],
#     figure_id: Optional[str] = None,
#     panel: str = "0,0",
#     labels: Optional[list[str]] = None,
#     colors: Optional[list[str]] = None,
#     xlabel: Optional[str] = None,
#     ylabel: Optional[str] = None,
#     title: Optional[str] = None,
# ) -> dict:
#     """Create a violin plot on specified panel."""
#     try:
#         fig, ax, fid = _get_axes(figure_id, panel)
# 
#         positions = list(range(1, len(data) + 1))
#         vp = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
# 
#         if colors is None:
#             colors = ["#007fbf", "#ff4433", "#14b514", "#c633ff", "#e25e33"]
#         for i, body in enumerate(vp["bodies"]):
#             body.set_facecolor(colors[i % len(colors)])
#             body.set_alpha(0.6)
# 
#         if labels:
#             ax.set_xticks(positions)
#             ax.set_xticklabels(labels)
#         if xlabel:
#             ax.set_xlabel(xlabel)
#         if ylabel:
#             ax.set_ylabel(ylabel)
#         if title:
#             ax.set_title(title)
# 
#         return {
#             "success": True,
#             "figure_id": fid,
#             "panel": panel,
#             "plot_type": "violin",
#             "n_groups": len(data),
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# __all__ = [
#     "plot_bar_handler",
#     "plot_scatter_handler",
#     "plot_line_handler",
#     "plot_box_handler",
#     "plot_violin_handler",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/_handlers_plot.py
# --------------------------------------------------------------------------------
