# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_core.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_core.py
# 
# """
# Core metadata collection orchestration.
# 
# This module provides the main collect_figure_metadata() function that orchestrates
# the collection of all metadata from matplotlib figures and axes.
# """
# 
# from typing import Dict
# import numpy as np
# 
# from scitex import logging
# 
# logger = logging.getLogger(__name__)
# 
# # Import sub-modules
# from ._figure_metadata import _initialize_metadata_structure, _collect_figure_metadata
# from ._axes_metadata import _collect_all_axes_metadata
# from ._dimensions import _collect_grid_info, _extract_figure_dimensions
# from ._style_parsing import _restructure_style, _collect_style_metadata, _extract_mode_and_method
# from ._plot_content import _detect_plot_type, _extract_artists, _extract_legend_info
# from ._data_linkage import _compute_csv_hash
# from ._precision import _round_metadata
# 
# 
# def collect_figure_metadata(fig, ax=None) -> Dict:
#     """
#     Collect all metadata from figure and axes for embedding in saved images.
# 
#     This function automatically extracts:
#     - Software versions (scitex, matplotlib)
#     - Timestamp
#     - Figure UUID (unique identifier)
#     - Figure/axes dimensions (mm, inch, px)
#     - DPI settings
#     - Margins
#     - Styling parameters (if available)
#     - Mode (display/publication)
#     - Creation method
#     - Plot type and axes information
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         Figure to collect metadata from
#     ax : matplotlib.axes.Axes, optional
#         Primary axes to collect dimension info from.
#         If not provided, uses first axes in figure.
# 
#     Returns
#     -------
#     dict
#         Complete metadata dictionary ready for embedding via scitex.io.embed_metadata()
# 
#     Examples
#     --------
#     >>> from scitex.plt.utils import create_axes_with_size_mm, collect_figure_metadata
#     >>> fig, ax = create_axes_with_size_mm(30, 21, mode='publication')
#     >>> ax.plot(x, y)
#     >>> metadata = collect_figure_metadata(fig, ax)
#     >>> print(metadata['figure']['size_mm'])
#     [30.00, 21.00]
# 
#     Notes
#     -----
#     This function is automatically called by FigWrapper.savefig() when
#     embed_metadata=True (the default). You typically don't need to call it manually.
# 
#     The collected metadata enables:
#     - Reproducing exact figure dimensions later
#     - Matching styling across multiple figures
#     - Documenting figure provenance
#     - Debugging dimension/DPI issues
#     """
#     # Initialize base metadata structure
#     metadata = _initialize_metadata_structure()
# 
#     # Collect grid information
#     all_axes, grid_shape = _collect_grid_info(ax) if ax is not None else ([], (1, 1))
# 
#     # Fallback to figure axes if no axes provided
#     if not all_axes and hasattr(fig, "axes") and len(fig.axes) > 0:
#         for idx, ax_item in enumerate(fig.axes):
#             all_axes.append((ax_item, 0, idx))
# 
#     # Collect figure-level metadata
#     if all_axes:
#         try:
#             metadata["figure"] = _extract_figure_dimensions(fig)
#         except Exception as e:
#             logger.warning(f"Could not extract figure dimension info: {e}")
# 
#     # Collect axes metadata
#     if all_axes:
#         try:
#             metadata["axes"] = _collect_all_axes_metadata(all_axes, fig, grid_shape)
#         except Exception as e:
#             logger.warning(f"Could not extract axes metadata: {e}")
# 
#     # Extract style metadata
#     try:
#         style_meta = _collect_style_metadata(fig, ax)
#         if style_meta:
#             metadata["style"] = style_meta
#     except Exception:
#         pass  # Style is optional
# 
#     # Extract mode and creation method
#     try:
#         mode, creation_method = _extract_mode_and_method(fig, ax)
#         if mode:
#             if "figure" not in metadata:
#                 metadata["figure"] = {}
#             metadata["figure"]["mode"] = mode
#         if creation_method:
#             metadata["runtime"]["created_with"] = creation_method
#     except Exception:
#         pass  # Mode/method are optional
# 
#     # Extract font information
#     try:
#         from .._get_actual_font import get_actual_font_name
#         actual_font = get_actual_font_name()
#         
#         if "style" in metadata:
#             if "global" not in metadata["style"]:
#                 metadata["style"]["global"] = {}
#             if "fonts" not in metadata["style"]["global"]:
#                 metadata["style"]["global"]["fonts"] = {}
#             
#             requested_font = metadata["style"]["global"]["fonts"].get("family", "Arial")
#             if "family" in metadata["style"]["global"]["fonts"]:
#                 del metadata["style"]["global"]["fonts"]["family"]
#             metadata["style"]["global"]["fonts"]["family_requested"] = requested_font
#             metadata["style"]["global"]["fonts"]["family_actual"] = actual_font
#             
#             if requested_font != actual_font:
#                 logger.warning(
#                     f"Font mismatch: Requested '{requested_font}' but using '{actual_font}'"
#                 )
#         else:
#             metadata["runtime"]["font_family_actual"] = actual_font
#     except Exception:
#         pass  # Font detection is optional
# 
#     # Extract plot content
#     if ax is not None:
#         try:
#             # Get primary axes for plot info
#             primary_ax = ax
#             if hasattr(ax, "_axes_scitex"):
#                 axes_array = ax._axes_scitex
#                 if isinstance(axes_array, np.ndarray) and axes_array.size > 0:
#                     primary_ax = axes_array.flat[0]
#                 else:
#                     primary_ax = axes_array
# 
#             # Find axes with history for plot type detection
#             ax_for_history = primary_ax
#             if not hasattr(primary_ax, 'history'):
#                 if hasattr(primary_ax, '_scitex_wrapper'):
#                     ax_for_history = primary_ax._scitex_wrapper
# 
#             # Extract plot info
#             plot_info = {}
#             
#             # Get axes position
#             ax_row, ax_col = 0, 0
#             if hasattr(primary_ax, "_scitex_metadata") and "position_in_grid" in primary_ax._scitex_metadata:
#                 pos = primary_ax._scitex_metadata["position_in_grid"]
#                 ax_row, ax_col = pos[0], pos[1]
#             plot_info["ax_id"] = f"ax_{ax_row:02d}_{ax_col:02d}"
# 
#             # Extract title
#             ax_mpl = primary_ax._axis_mpl if hasattr(primary_ax, '_axis_mpl') else primary_ax
#             title = ax_mpl.get_title()
#             if title:
#                 plot_info["title"] = title
# 
#             # Detect plot type
#             plot_type, method = _detect_plot_type(ax_for_history)
#             if plot_type:
#                 plot_info["type"] = plot_type
#             if method:
#                 plot_info["method"] = method
# 
#             if plot_info:
#                 metadata["plot"] = plot_info
# 
#             # Extract data hash
#             try:
#                 csv_hash = _compute_csv_hash(ax_for_history)
#                 if csv_hash:
#                     metadata["data"] = {"csv_hash": csv_hash}
#             except Exception:
#                 pass  # Data hash is optional
# 
#         except Exception as e:
#             logger.warning(f"Could not extract plot content: {e}")
# 
#     # Apply precision rounding
#     metadata = _round_metadata(metadata)
# 
#     return metadata

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_core.py
# --------------------------------------------------------------------------------
