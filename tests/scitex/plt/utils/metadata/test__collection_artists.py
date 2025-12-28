# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_collection_artists.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_collection_artists.py
# 
# """
# Collection artist extraction utilities.
# 
# This module provides functions to extract PathCollection (scatter), QuadMesh (hist2d),
# PolyCollection (hexbin/violin), and other collection artists from matplotlib axes.
# """
# 
# import matplotlib.colors as mcolors
# from ._label_parsing import _get_csv_column_names
# 
# 
# def _extract_scatter_artists(mpl_ax, ax_row, ax_col):
#     """
#     Extract PathCollection artists (scatter points).
# 
#     Parameters
#     ----------
#     mpl_ax : matplotlib.axes.Axes
#         Raw matplotlib axes
#     ax_row : int
#         Row position in grid
#     ax_col : int
#         Column position in grid
# 
#     Returns
#     -------
#     list
#         List of artist dictionaries
#     """
#     artists = []
# 
#     for i, coll in enumerate(mpl_ax.collections):
#         if "PathCollection" not in type(coll).__name__:
#             continue
# 
#         artist = {}
# 
#         scitex_id = getattr(coll, "_scitex_id", None)
#         label = coll.get_label()
# 
#         if scitex_id:
#             artist["id"] = scitex_id
#         elif label and not label.startswith("_"):
#             artist["id"] = label
#         else:
#             artist["id"] = f"scatter_{i}"
# 
#         artist["mark"] = "scatter"
# 
#         if label and not label.startswith("_"):
#             artist["label"] = label
#             artist["legend_included"] = True
#         else:
#             artist["legend_included"] = False
# 
#         artist["zorder"] = coll.get_zorder()
# 
#         # Backend layer
#         backend = {
#             "name": "matplotlib",
#             "artist_class": type(coll).__name__,
#             "props": {}
#         }
# 
#         try:
#             facecolors = coll.get_facecolor()
#             if len(facecolors) > 0:
#                 backend["props"]["facecolor"] = mcolors.to_hex(facecolors[0], keep_alpha=False)
#         except (ValueError, TypeError, IndexError):
#             pass
# 
#         try:
#             edgecolors = coll.get_edgecolor()
#             if len(edgecolors) > 0:
#                 backend["props"]["edgecolor"] = mcolors.to_hex(edgecolors[0], keep_alpha=False)
#         except (ValueError, TypeError, IndexError):
#             pass
# 
#         try:
#             sizes = coll.get_sizes()
#             if len(sizes) > 0:
#                 backend["props"]["size"] = float(sizes[0])
#         except (ValueError, TypeError, IndexError):
#             pass
# 
#         artist["backend"] = backend
# 
#         # data_ref
#         artist_id = artist.get("id", str(i))
#         artist["data_ref"] = _get_csv_column_names(artist_id, ax_row, ax_col)
# 
#         artists.append(artist)
# 
#     return artists
# 
# 
# def _extract_hist2d_hexbin_artists(mpl_ax, ax_for_detection, plot_type):
#     """
#     Extract QuadMesh (hist2d) and PolyCollection (hexbin) artists.
# 
#     Parameters
#     ----------
#     mpl_ax : matplotlib.axes.Axes
#         Raw matplotlib axes
#     ax_for_detection : axes wrapper
#         Axes wrapper with history
#     plot_type : str
#         Detected plot type
# 
#     Returns
#     -------
#     list
#         List of artist dictionaries
#     """
#     artists = []
# 
#     for i, coll in enumerate(mpl_ax.collections):
#         coll_type = type(coll).__name__
# 
#         if coll_type == "QuadMesh":
#             artist = {}
#             artist["id"] = f"hist2d_{i}"
#             artist["mark"] = "heatmap"
#             artist["role"] = "hist2d"
#             artist["legend_included"] = False
#             artist["zorder"] = coll.get_zorder()
# 
#             # Backend layer
#             backend = {
#                 "name": "matplotlib",
#                 "artist_class": coll_type,
#                 "props": {}
#             }
#             try:
#                 cmap = coll.get_cmap()
#                 if cmap:
#                     backend["props"]["cmap"] = cmap.name
#             except (ValueError, TypeError, AttributeError):
#                 pass
#             try:
#                 backend["props"]["vmin"] = float(coll.norm.vmin) if coll.norm else None
#                 backend["props"]["vmax"] = float(coll.norm.vmax) if coll.norm else None
#             except (ValueError, TypeError, AttributeError):
#                 pass
# 
#             artist["backend"] = backend
# 
#             # Extract hist2d result data
#             try:
#                 arr = coll.get_array()
#                 if arr is not None and len(arr) > 0:
#                     import numpy as np
#                     coords = coll.get_coordinates()
#                     if coords is not None and len(coords) > 0:
#                         n_ybins = coords.shape[0] - 1
#                         n_xbins = coords.shape[1] - 1
#                         xedges = coords[0, :, 0]
#                         yedges = coords[:, 0, 1]
# 
#                         artist["result"] = {
#                             "H_shape": [n_ybins, n_xbins],
#                             "n_xbins": int(n_xbins),
#                             "n_ybins": int(n_ybins),
#                             "xedges_range": [float(xedges[0]), float(xedges[-1])],
#                             "yedges_range": [float(yedges[0]), float(yedges[-1])],
#                             "count_range": [float(arr.min()), float(arr.max())],
#                             "total_count": int(arr.sum()),
#                         }
#             except (IndexError, TypeError, AttributeError, ValueError):
#                 pass
# 
#             artists.append(artist)
# 
#         elif coll_type == "PolyCollection":
#             arr = coll.get_array() if hasattr(coll, "get_array") else None
# 
#             # Check if this is hexbin
#             if arr is not None and len(arr) > 0 and plot_type == "hexbin":
#                 artist = {}
#                 artist["id"] = f"hexbin_{i}"
#                 artist["mark"] = "heatmap"
#                 artist["role"] = "hexbin"
#                 artist["legend_included"] = False
#                 artist["zorder"] = coll.get_zorder()
# 
#                 # Backend layer
#                 backend = {
#                     "name": "matplotlib",
#                     "artist_class": coll_type,
#                     "props": {}
#                 }
#                 try:
#                     cmap = coll.get_cmap()
#                     if cmap:
#                         backend["props"]["cmap"] = cmap.name
#                 except (ValueError, TypeError, AttributeError):
#                     pass
#                 try:
#                     backend["props"]["vmin"] = float(coll.norm.vmin) if coll.norm else None
#                     backend["props"]["vmax"] = float(coll.norm.vmax) if coll.norm else None
#                 except (ValueError, TypeError, AttributeError):
#                     pass
# 
#                 artist["backend"] = backend
# 
#                 # Add hexbin result info
#                 try:
#                     artist["result"] = {
#                         "n_hexagons": int(len(arr)),
#                         "count_range": [float(arr.min()), float(arr.max())] if len(arr) > 0 else None,
#                         "total_count": int(arr.sum()),
#                     }
#                 except (TypeError, AttributeError, ValueError):
#                     pass
# 
#                 artists.append(artist)
# 
#     return artists
# 
# 
# def _extract_violin_body_artists(mpl_ax, plot_type):
#     """
#     Extract violin body (PolyCollection) artists.
# 
#     Parameters
#     ----------
#     mpl_ax : matplotlib.axes.Axes
#         Raw matplotlib axes
#     plot_type : str
#         Detected plot type
# 
#     Returns
#     -------
#     list
#         List of artist dictionaries
#     """
#     artists = []
# 
#     if plot_type != "violin":
#         return artists
# 
#     for i, coll in enumerate(mpl_ax.collections):
#         coll_type = type(coll).__name__
# 
#         if coll_type == "PolyCollection" or (coll_type == "FillBetweenPolyCollection"):
#             # Check if this is a violin body (no array data)
#             arr = coll.get_array() if hasattr(coll, "get_array") else None
#             if arr is not None and len(arr) > 0:
#                 continue  # This is hexbin, not violin
# 
#             artist = {}
#             scitex_id = getattr(coll, "_scitex_id", None)
#             label = coll.get_label() if hasattr(coll, "get_label") else ""
# 
#             if scitex_id:
#                 artist["id"] = f"{scitex_id}_body_{i}"
#                 artist["group_id"] = scitex_id
#             else:
#                 artist["id"] = f"violin_body_{i}"
# 
#             artist["mark"] = "polygon"
#             artist["role"] = "violin_body"
#             artist["legend_included"] = False
#             artist["zorder"] = coll.get_zorder()
# 
#             # Backend layer
#             backend = {
#                 "name": "matplotlib",
#                 "artist_class": coll_type,
#                 "props": {}
#             }
#             try:
#                 facecolors = coll.get_facecolor()
#                 if len(facecolors) > 0:
#                     backend["props"]["facecolor"] = mcolors.to_hex(facecolors[0], keep_alpha=False)
#             except (ValueError, TypeError, IndexError):
#                 pass
#             try:
#                 edgecolors = coll.get_edgecolor()
#                 if len(edgecolors) > 0:
#                     backend["props"]["edgecolor"] = mcolors.to_hex(edgecolors[0], keep_alpha=False)
#             except (ValueError, TypeError, IndexError):
#                 pass
# 
#             artist["backend"] = backend
#             artists.append(artist)
# 
#     return artists

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_collection_artists.py
# --------------------------------------------------------------------------------
