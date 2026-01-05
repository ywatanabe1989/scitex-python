# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/_registry.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-08 23:30:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/_registry.py
# 
# """Plot registry organized by visualization purpose."""
# 
# import numpy as np
# 
# # Categories organized by visualization purpose
# CATEGORIES = {
#     "line": {
#         "name": "Line Plots",
#         "description": "Time series, trends, and continuous data",
#         "plots": [
#             "plot",
#             "step",
#             "stx_line",
#             "stx_shaded_line",
#         ],
#     },
#     "statistical": {
#         "name": "Statistical Summaries",
#         "description": "Mean, CI, std, and aggregated views",
#         "plots": [
#             "stx_mean_std",
#             "stx_mean_ci",
#             "stx_median_iqr",
#             "errorbar",
#             "stx_errorbar",
#         ],
#     },
#     "distribution": {
#         "name": "Distributions",
#         "description": "KDE, histograms, density estimation",
#         "plots": [
#             "hist",
#             "hist2d",
#             "stx_kde",
#             "stx_ecdf",
#             "stx_joyplot",
#         ],
#     },
#     "categorical": {
#         "name": "Categorical",
#         "description": "Bar charts, box plots, violin plots",
#         "plots": [
#             "bar",
#             "barh",
#             "stx_bar",
#             "stx_barh",
#             "boxplot",
#             "violinplot",
#             "stx_box",
#             "stx_violin",
#             "stx_boxplot",
#             "stx_violinplot",
#         ],
#     },
#     "scatter": {
#         "name": "Scatter & Points",
#         "description": "Point clouds, correlations",
#         "plots": [
#             "scatter",
#             "stx_scatter",
#             "stem",
#             "hexbin",
#         ],
#     },
#     "area": {
#         "name": "Area & Fill",
#         "description": "Filled regions, ranges",
#         "plots": [
#             "fill_between",
#             "fill_betweenx",
#             "stx_fill_between",
#             "stx_fillv",
#         ],
#     },
#     "grid": {
#         "name": "Grid & Matrix",
#         "description": "Heatmaps, images, confusion matrices",
#         "plots": [
#             "imshow",
#             "matshow",
#             "stx_imshow",
#             "stx_image",
#             "stx_heatmap",
#             "stx_conf_mat",
#         ],
#     },
#     "contour": {
#         "name": "Contours",
#         "description": "Contour lines and filled contours",
#         "plots": [
#             "contour",
#             "contourf",
#             "stx_contour",
#         ],
#     },
#     "vector": {
#         "name": "Vector Fields",
#         "description": "Quiver, streamlines",
#         "plots": [
#             "quiver",
#             "streamplot",
#         ],
#     },
#     "special": {
#         "name": "Special",
#         "description": "Pie charts, rasters, annotations",
#         "plots": [
#             "pie",
#             "stx_raster",
#             "stx_rectangle",
#         ],
#     },
# }
# 
# 
# def list_plots(category=None):
#     """List available plots.
# 
#     Parameters
#     ----------
#     category : str, optional
#         If provided, list plots in that category only.
# 
#     Returns
#     -------
#     dict or list
#         If category is None, returns dict of all categories.
#         If category is provided, returns list of plots in that category.
#     """
#     if category is None:
#         return {
#             cat: {
#                 "name": info["name"],
#                 "description": info["description"],
#                 "plots": info["plots"],
#             }
#             for cat, info in CATEGORIES.items()
#         }
# 
#     if category not in CATEGORIES:
#         raise ValueError(
#             f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}"
#         )
# 
#     return CATEGORIES[category]["plots"]
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/_registry.py
# --------------------------------------------------------------------------------
