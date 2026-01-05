# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_legend_extraction.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_legend_extraction.py
# 
# """
# Legend extraction utilities.
# 
# This module provides functions to extract legend information from matplotlib axes.
# """
# 
# from typing import Optional
# 
# 
# def _extract_legend_info(ax) -> Optional[dict]:
#     """
#     Extract legend information from axes.
# 
#     Uses matplotlib terminology for legend properties.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes to extract legend from
# 
#     Returns
#     -------
#     dict or None
#         Legend info dictionary with matplotlib properties, or None if no legend
#     """
#     legend = ax.get_legend()
#     if legend is None:
#         return None
# 
#     legend_info = {
#         "visible": legend.get_visible(),
#         "loc": legend._loc if hasattr(legend, "_loc") else "best",
#         "frameon": legend.get_frame_on() if hasattr(legend, "get_frame_on") else True,
#     }
# 
#     # ncol - number of columns
#     if hasattr(legend, "_ncols"):
#         legend_info["ncol"] = legend._ncols
#     elif hasattr(legend, "_ncol"):
#         legend_info["ncol"] = legend._ncol
# 
#     # Extract legend handles with artist references
#     # This allows reconstructing the legend by referencing artists
#     handles = []
#     texts = legend.get_texts()
#     legend_handles = legend.legend_handles if hasattr(legend, 'legend_handles') else []
# 
#     # Get the raw matplotlib axes for accessing lines to match IDs
#     mpl_ax = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax
# 
#     for i, text in enumerate(texts):
#         label_text = text.get_text()
#         handle_entry = {"label": label_text}
# 
#         # Try to get artist_id from corresponding handle
#         artist_id = None
#         if i < len(legend_handles):
#             handle = legend_handles[i]
#             # Check if handle has scitex_id
#             if hasattr(handle, "_scitex_id"):
#                 artist_id = handle._scitex_id
# 
#         # Fallback: find matching artist by label in axes artists
#         if artist_id is None:
#             # Check lines
#             for line in mpl_ax.lines:
#                 line_label = line.get_label()
#                 if line_label == label_text:
#                     if hasattr(line, "_scitex_id"):
#                         artist_id = line._scitex_id
#                     elif not line_label.startswith("_"):
#                         artist_id = line_label
#                     break
# 
#         # Check collections (scatter)
#         if artist_id is None:
#             for coll in mpl_ax.collections:
#                 coll_label = coll.get_label() if hasattr(coll, "get_label") else ""
#                 if coll_label == label_text:
#                     if hasattr(coll, "_scitex_id"):
#                         artist_id = coll._scitex_id
#                     elif coll_label and not coll_label.startswith("_"):
#                         artist_id = coll_label
#                     break
# 
#         # Check patches (bar/hist/pie)
#         if artist_id is None:
#             for patch in mpl_ax.patches:
#                 patch_label = patch.get_label() if hasattr(patch, "get_label") else ""
#                 if patch_label == label_text:
#                     if hasattr(patch, "_scitex_id"):
#                         artist_id = patch._scitex_id
#                     elif patch_label and not patch_label.startswith("_"):
#                         artist_id = patch_label
#                     break
# 
#         # Check images (imshow)
#         if artist_id is None:
#             for img in mpl_ax.images:
#                 img_label = img.get_label() if hasattr(img, "get_label") else ""
#                 if img_label == label_text:
#                     if hasattr(img, "_scitex_id"):
#                         artist_id = img._scitex_id
#                     elif img_label and not img_label.startswith("_"):
#                         artist_id = img_label
#                     break
# 
#         if artist_id:
#             handle_entry["artist_id"] = artist_id
# 
#         handles.append(handle_entry)
# 
#     if handles:
#         legend_info["handles"] = handles
# 
#     return legend_info

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_legend_extraction.py
# --------------------------------------------------------------------------------
