# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_image_text_artists.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_image_text_artists.py
# 
# """
# Image and text artist extraction utilities.
# 
# This module provides functions to extract AxesImage (imshow) and Text (annotations)
# artists from matplotlib axes.
# """
# 
# import matplotlib.colors as mcolors
# 
# 
# def _extract_image_artists(mpl_ax):
#     """
#     Extract AxesImage artists (imshow).
# 
#     Parameters
#     ----------
#     mpl_ax : matplotlib.axes.Axes
#         Raw matplotlib axes
# 
#     Returns
#     -------
#     list
#         List of artist dictionaries
#     """
#     artists = []
# 
#     for i, img in enumerate(mpl_ax.images):
#         img_type = type(img).__name__
# 
#         artist = {}
# 
#         scitex_id = getattr(img, "_scitex_id", None)
#         label = img.get_label() if hasattr(img, "get_label") else ""
# 
#         if scitex_id:
#             artist["id"] = scitex_id
#         elif label and not label.startswith("_"):
#             artist["id"] = label
#         else:
#             artist["id"] = f"image_{i}"
# 
#         artist["mark"] = "image"
#         artist["role"] = "image"
#         artist["legend_included"] = False
#         artist["zorder"] = img.get_zorder()
# 
#         # Backend layer
#         backend = {
#             "name": "matplotlib",
#             "artist_class": img_type,
#             "props": {}
#         }
#         try:
#             cmap = img.get_cmap()
#             if cmap:
#                 backend["props"]["cmap"] = cmap.name
#         except (ValueError, TypeError, AttributeError):
#             pass
#         try:
#             backend["props"]["vmin"] = float(img.norm.vmin) if img.norm else None
#             backend["props"]["vmax"] = float(img.norm.vmax) if img.norm else None
#         except (ValueError, TypeError, AttributeError):
#             pass
#         try:
#             backend["props"]["interpolation"] = img.get_interpolation()
#         except (ValueError, TypeError, AttributeError):
#             pass
# 
#         artist["backend"] = backend
#         artists.append(artist)
# 
#     return artists
# 
# 
# def _extract_text_artists(mpl_ax):
#     """
#     Extract Text artists (annotations, stats text, etc.).
# 
#     Parameters
#     ----------
#     mpl_ax : matplotlib.axes.Axes
#         Raw matplotlib axes
# 
#     Returns
#     -------
#     list
#         List of artist dictionaries
#     """
#     artists = []
#     text_count = 0
# 
#     for i, text_obj in enumerate(mpl_ax.texts):
#         text_content = text_obj.get_text()
#         if not text_content or text_content.strip() == "":
#             continue
# 
#         artist = {}
# 
#         scitex_id = getattr(text_obj, "_scitex_id", None)
# 
#         if scitex_id:
#             artist["id"] = scitex_id
#         else:
#             artist["id"] = f"text_{text_count}"
# 
#         artist["mark"] = "text"
# 
#         # Try to determine role from content or position
#         pos = text_obj.get_position()
#         # Check if this looks like stats annotation
#         if any(kw in text_content.lower() for kw in ['r=', 'p=', 'r²=', 'n=']):
#             artist["role"] = "stats_annotation"
#         else:
#             artist["role"] = "annotation"
# 
#         artist["legend_included"] = False
#         artist["zorder"] = text_obj.get_zorder()
# 
#         # Geometry - text position
#         artist["geometry"] = {
#             "x": pos[0],
#             "y": pos[1],
#         }
# 
#         artist["text"] = text_content
# 
#         # Backend layer
#         backend = {
#             "name": "matplotlib",
#             "artist_class": type(text_obj).__name__,
#             "props": {}
#         }
# 
#         try:
#             color = text_obj.get_color()
#             backend["props"]["color"] = mcolors.to_hex(color, keep_alpha=False)
#         except (ValueError, TypeError):
#             pass
# 
#         try:
#             backend["props"]["fontsize_pt"] = text_obj.get_fontsize()
#         except (ValueError, TypeError):
#             pass
# 
#         try:
#             backend["props"]["ha"] = text_obj.get_ha()
#             backend["props"]["va"] = text_obj.get_va()
#         except (ValueError, TypeError):
#             pass
# 
#         artist["backend"] = backend
# 
#         # data_ref for text position - only if explicitly tracked
#         if scitex_id:
#             artist["data_ref"] = {
#                 "x": f"text_{text_count}_x",
#                 "y": f"text_{text_count}_y",
#                 "content": f"text_{text_count}_content"
#             }
# 
#         text_count += 1
#         artists.append(artist)
# 
#     return artists

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_image_text_artists.py
# --------------------------------------------------------------------------------
