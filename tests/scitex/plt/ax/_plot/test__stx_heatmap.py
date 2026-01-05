# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_stx_heatmap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 13:00:00 (ywatanabe)"
# # File: ./src/scitex/plt/ax/_plot/_plot_heatmap.py
# 
# """Heatmap plotting with automatic annotation color switching."""
# 
# from typing import Any, List, Optional, Tuple, Union
# 
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.axes import Axes
# from matplotlib.colorbar import Colorbar
# from matplotlib.image import AxesImage
# 
# 
# def stx_heatmap(
#     ax: Union[Axes, "AxisWrapper"],
#     values_2d: np.ndarray,
#     x_labels: Optional[List[str]] = None,
#     y_labels: Optional[List[str]] = None,
#     cmap: str = "viridis",
#     cbar_label: str = "ColorBar Label",
#     annot_format: str = "{x:.1f}",
#     show_annot: bool = True,
#     annot_color_lighter: str = "black",
#     annot_color_darker: str = "white",
#     **kwargs: Any,
# ) -> Tuple[Union[Axes, "AxisWrapper"], AxesImage, Colorbar]:
#     """Plot a heatmap on the given axes with automatic annotation colors.
# 
#     Creates a heatmap visualization with optional cell annotations. Annotation
#     text colors are automatically switched based on background brightness for
#     optimal readability.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes or AxisWrapper
#         The axes to plot on.
#     values_2d : np.ndarray, shape (n_rows, n_cols)
#         2D array of data to display as heatmap.
#     x_labels : list of str, optional
#         Labels for the x-axis (columns).
#     y_labels : list of str, optional
#         Labels for the y-axis (rows).
#     cmap : str, default "viridis"
#         Colormap name to use.
#     cbar_label : str, default "ColorBar Label"
#         Label for the colorbar.
#     annot_format : str, default "{x:.1f}"
#         Format string for cell annotations.
#     show_annot : bool, default True
#         Whether to annotate the heatmap with values.
#     annot_color_lighter : str, default "black"
#         Text color for annotations on lighter backgrounds.
#     annot_color_darker : str, default "white"
#         Text color for annotations on darker backgrounds.
#     **kwargs : dict
#         Additional keyword arguments passed to imshow().
# 
#     Returns
#     -------
#     ax : matplotlib.axes.Axes or AxisWrapper
#         The axes with the heatmap.
#     im : matplotlib.image.AxesImage
#         The image object created by imshow.
#     cbar : matplotlib.colorbar.Colorbar
#         The colorbar object.
# 
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import scitex as stx
#     >>> data = np.random.rand(5, 10)
#     >>> fig, ax = stx.plt.subplots()
#     >>> ax, im, cbar = stx.plt.ax.stx_heatmap(
#     ...     ax, data,
#     ...     x_labels=[f"X{i}" for i in range(10)],
#     ...     y_labels=[f"Y{i}" for i in range(5)],
#     ...     cmap="Blues"
#     ... )
#     """
# 
#     im, cbar = _mpl_heatmap(
#         values_2d,
#         x_labels,
#         y_labels,
#         ax=ax,
#         cmap=cmap,
#         cbarlabel=cbar_label,
#     )
# 
#     if show_annot:
#         textcolors = _switch_annot_colors(cmap, annot_color_lighter, annot_color_darker)
#         texts = _mpl_annotate_heatmap(
#             im,
#             valfmt=annot_format,
#             textcolors=textcolors,
#         )
# 
#     return ax, im, cbar
# 
# 
# def _switch_annot_colors(
#     cmap: str,
#     annot_color_lighter: str,
#     annot_color_darker: str,
# ) -> Tuple[str, str]:
#     """Determine annotation text colors based on colormap brightness.
# 
#     Uses perceived brightness (ITU-R BT.709) to select appropriate text
#     colors for light vs dark backgrounds in the colormap.
# 
#     Parameters
#     ----------
#     cmap : str
#         Colormap name.
#     annot_color_lighter : str
#         Color to use on lighter backgrounds.
#     annot_color_darker : str
#         Color to use on darker backgrounds.
# 
#     Returns
#     -------
#     tuple of str
#         (color_for_dark_bg, color_for_light_bg) text colors.
#     """
#     cmap_obj = plt.cm.get_cmap(cmap)
# 
#     # Sample colormap at extremes (avoiding edge effects)
#     dark_color = cmap_obj(0.1)
#     light_color = cmap_obj(0.9)
# 
#     # Calculate perceived brightness using ITU-R BT.709 coefficients
#     dark_brightness = (
#         0.2126 * dark_color[0] + 0.7152 * dark_color[1] + 0.0722 * dark_color[2]
#     )
# 
#     # Choose text colors based on background brightness
#     if dark_brightness < 0.5:
#         return (annot_color_lighter, annot_color_darker)
#     else:
#         return (annot_color_darker, annot_color_lighter)
# 
# 
# def _mpl_heatmap(
#     data: np.ndarray,
#     row_labels: Optional[List[str]],
#     col_labels: Optional[List[str]],
#     ax: Optional[Axes] = None,
#     cbar_kw: Optional[dict] = None,
#     cbarlabel: str = "",
#     **kwargs: Any,
# ) -> Tuple[AxesImage, Colorbar]:
#     """Create a heatmap with imshow and add a colorbar.
# 
#     Parameters
#     ----------
#     data : np.ndarray
#         2D array of data to display.
#     row_labels : list of str or None
#         Labels for the rows (y-axis).
#     col_labels : list of str or None
#         Labels for the columns (x-axis).
#     ax : matplotlib.axes.Axes, optional
#         Axes to plot on. If None, uses current axes.
#     cbar_kw : dict, optional
#         Keyword arguments for colorbar creation.
#     cbarlabel : str, default ""
#         Label for the colorbar.
#     **kwargs : dict
#         Additional keyword arguments passed to imshow().
# 
#     Returns
#     -------
#     im : matplotlib.image.AxesImage
#         The image object.
#     cbar : matplotlib.colorbar.Colorbar
#         The colorbar object.
#     """
# 
#     if ax is None:
#         ax = plt.gca()
# 
#     if cbar_kw is None:
#         cbar_kw = {}
# 
#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)
# 
#     # Create colorbar with proper formatting
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
# 
#     # Set colorbar border width to match axes spines
#     cbar.outline.set_linewidth(0.2 * 2.83465)  # 0.2mm in points
# 
#     # Format colorbar ticks
#     from matplotlib.ticker import MaxNLocator
# 
#     cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
#     cbar.ax.tick_params(width=0.2 * 2.83465, length=0.8 * 2.83465)  # Match tick styling
# 
#     # Show all ticks and label them with the respective list entries.
#     ax.set_xticks(
#         range(data.shape[1]),
#         labels=col_labels,
#         # rotation=45,
#         # ha="right",
#         # rotation_mode="anchor",
#     )
#     ax.set_yticks(range(data.shape[0]), labels=row_labels)
# 
#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
# 
#     # Show all 4 spines for heatmap
#     ax.spines[:].set_visible(True)
# 
#     # Set aspect ratio to 'equal' for square cells (1:1)
#     ax.set_aspect("equal", adjustable="box")
# 
#     ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
#     ax.tick_params(which="minor", bottom=False, left=False)
# 
#     return im, cbar
# 
# 
# def _calc_annot_fontsize(n_rows: int, n_cols: int) -> float:
#     """Calculate dynamic annotation font size based on cell count.
# 
#     Uses a base size of 6pt for small heatmaps and scales down for larger ones.
# 
#     Parameters
#     ----------
#     n_rows : int
#         Number of rows in the heatmap.
#     n_cols : int
#         Number of columns in the heatmap.
# 
#     Returns
#     -------
#     float
#         Font size in points.
#     """
#     # Base font size for small heatmaps (e.g., 5x5)
#     BASE_FONTSIZE = 6.0
#     BASE_CELLS = 5  # Reference dimension
# 
#     # Use the larger dimension to scale
#     max_dim = max(n_rows, n_cols)
# 
#     if max_dim <= BASE_CELLS:
#         return BASE_FONTSIZE
#     elif max_dim <= 10:
#         # Linear interpolation: 6pt at 5 cells, 5pt at 10 cells
#         return BASE_FONTSIZE - (max_dim - BASE_CELLS) * 0.2
#     elif max_dim <= 20:
#         # 5pt at 10 cells, 4pt at 20 cells
#         return 5.0 - (max_dim - 10) * 0.1
#     else:
#         # Minimum 3pt for very large heatmaps
#         return max(3.0, 4.0 - (max_dim - 20) * 0.05)
# 
# 
# def _mpl_annotate_heatmap(
#     im: AxesImage,
#     data: Optional[np.ndarray] = None,
#     valfmt: str = "{x:.2f}",
#     textcolors: Tuple[str, str] = ("lightgray", "black"),
#     threshold: Optional[float] = None,
#     fontsize: Optional[float] = None,
#     **textkw: Any,
# ) -> List:
#     """Annotate a heatmap with cell values.
# 
#     Parameters
#     ----------
#     im : matplotlib.image.AxesImage
#         The image to be annotated.
#     data : np.ndarray, optional
#         Data used to annotate. If None, uses the image's array.
#     valfmt : str, default "{x:.2f}"
#         Format string for the annotations.
#     textcolors : tuple of str, default ("lightgray", "black")
#         Colors for annotations. First color for values below threshold,
#         second for values above.
#     threshold : float, optional
#         Value in normalized colormap space (0 to 1) above which the
#         second color is used. If None, uses 0.7 * max(data).
#     fontsize : float, optional
#         Font size in points. If None, dynamically calculated based on
#         cell count (6pt base, scaling down for larger heatmaps).
#     **textkw : dict
#         Additional keyword arguments passed to ax.text().
# 
#     Returns
#     -------
#     texts : list of matplotlib.text.Text
#         The annotation text objects.
#     """
# 
#     if not isinstance(data, (list, np.ndarray)):
#         data = im.get_array()
# 
#     # Calculate dynamic font size if not specified
#     if fontsize is None:
#         fontsize = _calc_annot_fontsize(data.shape[0], data.shape[1])
# 
#     # Normalize the threshold to the images color range.
#     if threshold is not None:
#         threshold = im.norm(threshold)
#     else:
#         # Use 0.7 instead of 0.5 for better visibility with most colormaps
#         threshold = im.norm(data.max()) * 0.7
# 
#     # Set default alignment to center, but allow it to be
#     # overwritten by textkw.
#     kw = dict(
#         horizontalalignment="center", verticalalignment="center", fontsize=fontsize
#     )
#     kw.update(textkw)
# 
#     # Get the formatter in case a string is supplied
#     if isinstance(valfmt, str):
#         valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
# 
#     # Loop over the data and create a `Text` for each "pixel".
#     # Change the text's color depending on the data.
#     texts = []
#     for ii in range(data.shape[0]):
#         for jj in range(data.shape[1]):
#             kw.update(color=textcolors[int(im.norm(data[ii, jj]) > threshold)])
#             text = im.axes.text(jj, ii, valfmt(data[ii, jj], None), **kw)
#             texts.append(text)
# 
#     return texts
# 
# 
# if __name__ == "__main__":
#     import matplotlib
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     import numpy as np
# 
#     data = np.random.rand(5, 10)
#     x_labels = [f"X{ii + 1}" for ii in range(5)]
#     y_labels = [f"Y{ii + 1}" for ii in range(10)]
# 
#     fig, ax = plt.subplots()
# 
#     im, cbar = stx_heatmap(
#         ax,
#         data,
#         x_labels=x_labels,
#         y_labels=y_labels,
#         show_annot=True,
#         annot_color_lighter="white",
#         annot_color_darker="black",
#         cmap="Blues",
#     )
# 
#     fig.tight_layout()
#     plt.show()
#     # EOF
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_plot/_stx_heatmap.py
# --------------------------------------------------------------------------------
