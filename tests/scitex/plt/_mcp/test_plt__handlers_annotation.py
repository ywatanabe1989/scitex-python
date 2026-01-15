# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/_handlers_annotation.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-13
# # File: src/scitex/plt/_mcp/_handlers_annotation.py
# 
# """Annotation MCP handlers for SciTeX plt module."""
# 
# from __future__ import annotations
# 
# from typing import Optional
# 
# from ._handlers_figure import _get_axes
# 
# 
# async def add_significance_handler(
#     x1: float,
#     x2: float,
#     y: float,
#     text: str,
#     figure_id: Optional[str] = None,
#     panel: str = "0,0",
#     height: Optional[float] = None,
# ) -> dict:
#     """Add significance bracket between two groups."""
#     try:
#         from scitex.plt.styles.presets import SCITEX_STYLE
#         from scitex.plt.utils import mm_to_pt
# 
#         fig, ax, fid = _get_axes(figure_id, panel)
# 
#         ylim = ax.get_ylim()
#         ylim_range = float(ylim[1]) - float(ylim[0])
#         h = height if height else 0.1 * ylim_range
# 
#         # Use SCITEX line width (0.2mm converted to points)
#         line_width_mm = SCITEX_STYLE.get("trace_thickness_mm", 0.2)
#         line_width_pt = mm_to_pt(line_width_mm)
# 
#         # Draw bracket with SCITEX styling
#         ax.plot(
#             [x1, x1, x2, x2],
#             [y, y + h, y + h, y],
#             color="black",
#             linewidth=line_width_pt,
#         )
# 
#         ax.text(
#             (x1 + x2) / 2,
#             y + h + 0.02 * ylim_range,
#             text,
#             ha="center",
#             va="bottom",
#             fontsize=6,
#         )
# 
#         return {
#             "success": True,
#             "figure_id": fid,
#             "panel": panel,
#             "bracket": {"x1": x1, "x2": x2, "y": y, "text": text},
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def add_panel_label_handler(
#     label: str,
#     figure_id: Optional[str] = None,
#     panel: str = "0,0",
#     x: float = -0.15,
#     y: float = 1.1,
#     fontsize: float = 10,
#     fontweight: str = "bold",
# ) -> dict:
#     """Add panel label (A, B, C, etc.) to a panel."""
#     try:
#         fig, ax, fid = _get_axes(figure_id, panel)
# 
#         ax.text(
#             x,
#             y,
#             label,
#             transform=ax.transAxes,
#             fontsize=fontsize,
#             fontweight=fontweight,
#         )
# 
#         return {
#             "success": True,
#             "figure_id": fid,
#             "panel": panel,
#             "label": label,
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# __all__ = [
#     "add_significance_handler",
#     "add_panel_label_handler",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/_handlers_annotation.py
# --------------------------------------------------------------------------------
