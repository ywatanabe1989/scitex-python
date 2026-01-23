# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/_handlers_figure.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-13
# # File: src/scitex/plt/_mcp/_handlers_figure.py
# 
# """Figure management MCP handlers for SciTeX plt module."""
# 
# from __future__ import annotations
# 
# import asyncio
# from typing import Any, Optional
# 
# # Figure registry for tracking active figures across MCP calls
# _FIGURE_REGISTRY: dict[str, dict[str, Any]] = {}
# 
# 
# def _get_axes(figure_id: Optional[str], panel: str):
#     """Helper to get axes from figure registry."""
#     if figure_id is None:
#         if not _FIGURE_REGISTRY:
#             raise ValueError("No active figures. Call create_figure first.")
#         figure_id = list(_FIGURE_REGISTRY.keys())[-1]
# 
#     if figure_id not in _FIGURE_REGISTRY:
#         raise ValueError(f"Figure '{figure_id}' not found")
# 
#     fig_data = _FIGURE_REGISTRY[figure_id]
#     axes = fig_data["axes"]
# 
#     if "," in panel:
#         row, col = map(int, panel.split(","))
#         try:
#             ax = axes[row, col]
#         except (TypeError, IndexError):
#             ax = axes
#     else:
#         idx = ord(panel.upper()) - ord("A")
#         if hasattr(axes, "flat"):
#             ax = list(axes.flat)[idx]
#         elif hasattr(axes, "__getitem__"):
#             ax = axes[idx]
#         else:
#             ax = axes
# 
#     return fig_data["fig"], ax, figure_id
# 
# 
# async def create_figure_handler(
#     nrows: int = 1,
#     ncols: int = 1,
#     axes_width_mm: Optional[float] = None,
#     axes_height_mm: Optional[float] = None,
#     space_w_mm: Optional[float] = None,
#     space_h_mm: Optional[float] = None,
# ) -> dict:
#     """Create a multi-panel figure canvas with SciTeX style."""
#     import uuid
# 
#     try:
#         import scitex.plt as splt
# 
#         kwargs = {"nrows": nrows, "ncols": ncols}
#         if axes_width_mm is not None:
#             kwargs["axes_width_mm"] = axes_width_mm
#         if axes_height_mm is not None:
#             kwargs["axes_height_mm"] = axes_height_mm
#         if space_w_mm is not None:
#             kwargs["space_w_mm"] = space_w_mm
#         if space_h_mm is not None:
#             kwargs["space_h_mm"] = space_h_mm
# 
#         fig, axes = splt.subplots(**kwargs)
# 
#         figure_id = str(uuid.uuid4())[:8]
# 
#         _FIGURE_REGISTRY[figure_id] = {
#             "fig": fig,
#             "axes": axes,
#             "nrows": nrows,
#             "ncols": ncols,
#         }
# 
#         return {
#             "success": True,
#             "figure_id": figure_id,
#             "nrows": nrows,
#             "ncols": ncols,
#             "message": f"Created {nrows}x{ncols} figure with SciTeX style",
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def crop_figure_handler(
#     input_path: str,
#     output_path: Optional[str] = None,
#     margin: int = 12,
#     overwrite: bool = False,
# ) -> dict:
#     """Auto-crop whitespace from a saved figure image."""
#     try:
#         from scitex.plt import crop
# 
#         loop = asyncio.get_event_loop()
#         result_path = await loop.run_in_executor(
#             None,
#             lambda: crop(
#                 input_path=input_path,
#                 output_path=output_path,
#                 margin=margin,
#                 overwrite=overwrite,
#             ),
#         )
# 
#         return {
#             "success": True,
#             "input_path": input_path,
#             "output_path": str(result_path),
#             "margin_pixels": margin,
#             "message": f"Cropped figure saved to {result_path}",
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def save_figure_handler(
#     output_path: str,
#     figure_id: Optional[str] = None,
#     dpi: int = 300,
#     crop: bool = True,
# ) -> dict:
#     """Save the figure to file."""
#     try:
#         if figure_id is None:
#             if not _FIGURE_REGISTRY:
#                 raise ValueError("No active figures")
#             figure_id = list(_FIGURE_REGISTRY.keys())[-1]
# 
#         if figure_id not in _FIGURE_REGISTRY:
#             raise ValueError(f"Figure '{figure_id}' not found")
# 
#         fig = _FIGURE_REGISTRY[figure_id]["fig"]
# 
#         fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
# 
#         final_path = output_path
#         if crop and output_path.endswith(".png"):
#             from scitex.plt import crop as crop_fn
# 
#             final_path = crop_fn(output_path, overwrite=True)
# 
#         return {
#             "success": True,
#             "figure_id": figure_id,
#             "output_path": str(final_path),
#             "dpi": dpi,
#             "cropped": crop,
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# async def close_figure_handler(figure_id: Optional[str] = None) -> dict:
#     """Close a figure and free memory."""
#     try:
#         import scitex.plt as splt
# 
#         if figure_id is None:
#             if not _FIGURE_REGISTRY:
#                 return {"success": True, "message": "No figures to close"}
#             figure_id = list(_FIGURE_REGISTRY.keys())[-1]
# 
#         if figure_id in _FIGURE_REGISTRY:
#             fig = _FIGURE_REGISTRY[figure_id]["fig"]
#             splt.close(fig)
#             del _FIGURE_REGISTRY[figure_id]
# 
#         return {
#             "success": True,
#             "figure_id": figure_id,
#             "message": f"Closed figure {figure_id}",
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
# 
# 
# __all__ = [
#     "_FIGURE_REGISTRY",
#     "_get_axes",
#     "create_figure_handler",
#     "crop_figure_handler",
#     "save_figure_handler",
#     "close_figure_handler",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/_handlers_figure.py
# --------------------------------------------------------------------------------
