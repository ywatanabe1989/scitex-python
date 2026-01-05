# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_mm_layout.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# """Millimeter-based layout control for matplotlib figures."""
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# from ._AxesWrapper import AxesWrapper
# from ._AxisWrapper import AxisWrapper
# from ._FigWrapper import FigWrapper
# 
# 
# def create_with_mm_control(
#     *args,
#     track=True,
#     sharex=False,
#     sharey=False,
#     axes_width_mm=None,
#     axes_height_mm=None,
#     margin_left_mm=None,
#     margin_right_mm=None,
#     margin_bottom_mm=None,
#     margin_top_mm=None,
#     space_w_mm=None,
#     space_h_mm=None,
#     axes_thickness_mm=None,
#     tick_length_mm=None,
#     tick_thickness_mm=None,
#     trace_thickness_mm=None,
#     marker_size_mm=None,
#     axis_font_size_pt=None,
#     tick_font_size_pt=None,
#     title_font_size_pt=None,
#     legend_font_size_pt=None,
#     suptitle_font_size_pt=None,
#     label_pad_pt=None,
#     tick_pad_pt=None,
#     title_pad_pt=None,
#     font_family=None,
#     n_ticks=None,
#     mode=None,
#     dpi=None,
#     styles=None,
#     transparent=None,
#     theme=None,
#     **kwargs,
# ):
#     """Create figure with mm-based control over axes dimensions.
# 
#     Returns
#     -------
#     tuple
#         (FigWrapper, AxisWrapper or AxesWrapper)
#     """
#     from scitex.plt.utils import apply_style_mm, mm_to_inch
# 
#     # Parse nrows, ncols from args or kwargs
#     nrows, ncols = 1, 1
#     if len(args) >= 1:
#         nrows = args[0]
#     elif "nrows" in kwargs:
#         nrows = kwargs.pop("nrows")
#     if len(args) >= 2:
#         ncols = args[1]
#     elif "ncols" in kwargs:
#         ncols = kwargs.pop("ncols")
# 
#     n_axes = nrows * ncols
# 
#     # Apply mode-specific defaults
#     if mode == "display":
#         scale_factor = 3.0
#         dpi = dpi or 100
#     else:
#         scale_factor = 1.0
#         dpi = dpi or 300
# 
#     # Set defaults with scaling
#     if axes_width_mm is None:
#         axes_width_mm = 30.0 * scale_factor
#     elif mode == "display":
#         axes_width_mm = axes_width_mm * scale_factor
# 
#     if axes_height_mm is None:
#         axes_height_mm = 21.0 * scale_factor
#     elif mode == "display":
#         axes_height_mm = axes_height_mm * scale_factor
# 
#     margin_left_mm = (
#         margin_left_mm if margin_left_mm is not None else (5.0 * scale_factor)
#     )
#     margin_right_mm = (
#         margin_right_mm if margin_right_mm is not None else (2.0 * scale_factor)
#     )
#     margin_bottom_mm = (
#         margin_bottom_mm if margin_bottom_mm is not None else (5.0 * scale_factor)
#     )
#     margin_top_mm = margin_top_mm if margin_top_mm is not None else (2.0 * scale_factor)
#     space_w_mm = space_w_mm if space_w_mm is not None else (3.0 * scale_factor)
#     space_h_mm = space_h_mm if space_h_mm is not None else (3.0 * scale_factor)
# 
#     # Handle list vs scalar for axes dimensions
#     if isinstance(axes_width_mm, (list, tuple)):
#         ax_widths_mm = list(axes_width_mm)
#         if len(ax_widths_mm) != n_axes:
#             raise ValueError(
#                 f"axes_width_mm list length ({len(ax_widths_mm)}) "
#                 f"must match nrows*ncols ({n_axes})"
#             )
#     else:
#         ax_widths_mm = [axes_width_mm] * n_axes
# 
#     if isinstance(axes_height_mm, (list, tuple)):
#         ax_heights_mm = list(axes_height_mm)
#         if len(ax_heights_mm) != n_axes:
#             raise ValueError(
#                 f"axes_height_mm list length ({len(ax_heights_mm)}) "
#                 f"must match nrows*ncols ({n_axes})"
#             )
#     else:
#         ax_heights_mm = [axes_height_mm] * n_axes
# 
#     # Calculate figure size from axes grid
#     ax_widths_2d = np.array(ax_widths_mm).reshape(nrows, ncols)
#     ax_heights_2d = np.array(ax_heights_mm).reshape(nrows, ncols)
# 
#     max_widths_per_col = ax_widths_2d.max(axis=0)
#     max_heights_per_row = ax_heights_2d.max(axis=1)
# 
#     total_width_mm = (
#         margin_left_mm
#         + max_widths_per_col.sum()
#         + (ncols - 1) * space_w_mm
#         + margin_right_mm
#     )
#     total_height_mm = (
#         margin_bottom_mm
#         + max_heights_per_row.sum()
#         + (nrows - 1) * space_h_mm
#         + margin_top_mm
#     )
# 
#     # Create figure
#     figsize_inch = (mm_to_inch(total_width_mm), mm_to_inch(total_height_mm))
#     if transparent:
#         fig_mpl = plt.figure(figsize=figsize_inch, dpi=dpi, facecolor="none")
#     else:
#         fig_mpl = plt.figure(figsize=figsize_inch, dpi=dpi)
# 
#     # Store theme on figure
#     if theme is not None:
#         fig_mpl._scitex_theme = theme
# 
#     # Create axes array and position each one manually
#     axes_mpl_list = []
#     ax_idx = 0
# 
#     for row in range(nrows):
#         for col in range(ncols):
#             # Calculate position
#             left_mm = margin_left_mm + max_widths_per_col[:col].sum() + col * space_w_mm
#             rows_below = nrows - row - 1
#             bottom_mm = (
#                 margin_bottom_mm
#                 + max_heights_per_row[row + 1 :].sum()
#                 + rows_below * space_h_mm
#             )
# 
#             # Convert to figure coordinates [0-1]
#             left = left_mm / total_width_mm
#             bottom = bottom_mm / total_height_mm
#             width = ax_widths_mm[ax_idx] / total_width_mm
#             height = ax_heights_mm[ax_idx] / total_height_mm
# 
#             # Create axes
#             ax_mpl = fig_mpl.add_axes([left, bottom, width, height])
#             if transparent:
#                 ax_mpl.patch.set_alpha(0.0)
#             axes_mpl_list.append(ax_mpl)
# 
#             # Tag with metadata
#             ax_mpl._scitex_metadata = {
#                 "created_with": "scitex.plt.subplots",
#                 "mode": mode or "publication",
#                 "axes_size_mm": (ax_widths_mm[ax_idx], ax_heights_mm[ax_idx]),
#                 "position_in_grid": (row, col),
#             }
#             ax_idx += 1
# 
#     # Apply styling to each axes
#     suptitle_font_size_pt_value = None
#     for i, ax_mpl in enumerate(axes_mpl_list):
#         # Determine which style dict to use
#         if styles is not None:
#             if isinstance(styles, list):
#                 if len(styles) != n_axes:
#                     raise ValueError(
#                         f"styles list length ({len(styles)}) "
#                         f"must match nrows*ncols ({n_axes})"
#                     )
#                 style_dict = styles[i]
#             else:
#                 style_dict = styles
#         else:
#             # Build style dict from individual parameters
#             style_dict = {}
#             if axes_thickness_mm is not None:
#                 style_dict["axis_thickness_mm"] = axes_thickness_mm
#             if tick_length_mm is not None:
#                 style_dict["tick_length_mm"] = tick_length_mm
#             if tick_thickness_mm is not None:
#                 style_dict["tick_thickness_mm"] = tick_thickness_mm
#             if trace_thickness_mm is not None:
#                 style_dict["trace_thickness_mm"] = trace_thickness_mm
#             if marker_size_mm is not None:
#                 style_dict["marker_size_mm"] = marker_size_mm
#             if axis_font_size_pt is not None:
#                 style_dict["axis_font_size_pt"] = axis_font_size_pt
#             if tick_font_size_pt is not None:
#                 style_dict["tick_font_size_pt"] = tick_font_size_pt
#             if title_font_size_pt is not None:
#                 style_dict["title_font_size_pt"] = title_font_size_pt
#             if legend_font_size_pt is not None:
#                 style_dict["legend_font_size_pt"] = legend_font_size_pt
#             if suptitle_font_size_pt is not None:
#                 style_dict["suptitle_font_size_pt"] = suptitle_font_size_pt
#             if label_pad_pt is not None:
#                 style_dict["label_pad_pt"] = label_pad_pt
#             if tick_pad_pt is not None:
#                 style_dict["tick_pad_pt"] = tick_pad_pt
#             if title_pad_pt is not None:
#                 style_dict["title_pad_pt"] = title_pad_pt
#             if font_family is not None:
#                 style_dict["font_family"] = font_family
#             if n_ticks is not None:
#                 style_dict["n_ticks"] = n_ticks
# 
#         # Always add theme to style_dict
#         if theme is not None:
#             style_dict["theme"] = theme
# 
#         # Extract suptitle font size if available
#         if "suptitle_font_size_pt" in style_dict:
#             suptitle_font_size_pt_value = style_dict["suptitle_font_size_pt"]
# 
#         # Apply style if not empty
#         if style_dict:
#             apply_style_mm(ax_mpl, style_dict)
#             ax_mpl._scitex_metadata["style_mm"] = style_dict
# 
#     # Store suptitle font size in figure metadata
#     if suptitle_font_size_pt_value is not None:
#         fig_mpl._scitex_suptitle_font_size_pt = suptitle_font_size_pt_value
# 
#     # Wrap the figure
#     fig_scitex = FigWrapper(fig_mpl)
# 
#     # Reshape axes list
#     axes_array_mpl = np.array(axes_mpl_list).reshape(nrows, ncols)
# 
#     # Handle single axis case
#     if n_axes == 1:
#         ax_mpl_scalar = axes_array_mpl.item()
#         axis_scitex = AxisWrapper(fig_scitex, ax_mpl_scalar, track)
#         fig_scitex.axes = [axis_scitex]
#         ax_mpl_scalar._scitex_wrapper = axis_scitex
#         return fig_scitex, axis_scitex
# 
#     # Handle multiple axes case
#     axes_flat_scitex_list = []
#     for ax_mpl in axes_mpl_list:
#         ax_scitex = AxisWrapper(fig_scitex, ax_mpl, track)
#         ax_mpl._scitex_wrapper = ax_scitex
#         axes_flat_scitex_list.append(ax_scitex)
# 
#     axes_array_scitex = np.array(axes_flat_scitex_list).reshape(nrows, ncols)
#     axes_scitex = AxesWrapper(fig_scitex, axes_array_scitex)
#     fig_scitex.axes = axes_scitex
# 
#     return fig_scitex, axes_scitex
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_mm_layout.py
# --------------------------------------------------------------------------------
