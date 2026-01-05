# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_dimension_viewer.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-19 12:30:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_dimension_viewer.py
# 
# """
# Visual dimension viewer/debugger for matplotlib figures.
# 
# This tool helps you understand and debug the relationship between:
# - Millimeters (mm)
# - Inches
# - Pixels (px)
# - DPI (dots per inch)
# - Figure size vs Axes size
# 
# Very useful when you're confused about dimensions!
# """
# 
# __FILE__ = __file__
# 
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import numpy as np
# 
# 
# def view_dimensions(fig, ax, show_rulers=True, show_grid=True, output_path=None):
#     """
#     Create a visual representation of figure/axes dimensions.
# 
#     This creates a separate diagnostic figure showing:
#     - The figure canvas (gray)
#     - The axes box (white with border)
#     - Dimension annotations
#     - Rulers with mm/pixel markings
#     - Grid overlay
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         Figure to analyze
#     ax : matplotlib.axes.Axes
#         Axes to analyze
#     show_rulers : bool, optional
#         Show rulers with mm/pixel markings (default: True)
#     show_grid : bool, optional
#         Show grid overlay (default: True)
#     output_path : str, optional
#         If provided, save the diagnostic figure to this path
# 
#     Returns
#     -------
#     fig_diag : matplotlib.figure.Figure
#         The diagnostic figure showing dimensions
# 
#     Examples
#     --------
#     >>> from scitex.plt.utils import create_axes_with_size_mm, view_dimensions
#     >>> fig, ax = create_axes_with_size_mm(30, 21, mode='publication')
#     >>> ax.plot([0, 1], [0, 1])
#     >>> fig_diag = view_dimensions(fig, ax, output_path='/tmp/dimensions.png')
#     >>> plt.show()
#     """
#     from ._figure_from_axes_mm import get_dimension_info
# 
#     # Get dimension info
#     info = get_dimension_info(fig, ax)
# 
#     # Create diagnostic figure (larger for visibility)
#     fig_diag = plt.figure(figsize=(10, 8), facecolor="white")
#     ax_diag = fig_diag.add_subplot(111)
# 
#     # Get dimensions
#     fig_w_mm, fig_h_mm = info["figure_size_mm"]
#     axes_w_mm, axes_h_mm = info["axes_size_mm"]
#     axes_pos = info["axes_position"]
#     dpi = info["dpi"]
# 
#     # Calculate margins in mm
#     margin_left_mm = axes_pos[0] * fig_w_mm
#     margin_bottom_mm = axes_pos[1] * fig_h_mm
#     margin_right_mm = fig_w_mm - (margin_left_mm + axes_w_mm)
#     margin_top_mm = fig_h_mm - (margin_bottom_mm + axes_h_mm)
# 
#     # Draw figure canvas (gray rectangle)
#     fig_rect = mpatches.Rectangle(
#         (0, 0),
#         fig_w_mm,
#         fig_h_mm,
#         linewidth=2,
#         edgecolor="black",
#         facecolor="lightgray",
#         alpha=0.3,
#         label="Figure canvas",
#     )
#     ax_diag.add_patch(fig_rect)
# 
#     # Draw axes box (white rectangle)
#     axes_rect = mpatches.Rectangle(
#         (margin_left_mm, margin_bottom_mm),
#         axes_w_mm,
#         axes_h_mm,
#         linewidth=3,
#         edgecolor="blue",
#         facecolor="white",
#         alpha=0.8,
#         label="Axes box",
#     )
#     ax_diag.add_patch(axes_rect)
# 
#     # Annotate dimensions
#     # Figure dimensions
#     ax_diag.annotate(
#         "",
#         xy=(fig_w_mm, -3),
#         xytext=(0, -3),
#         arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
#     )
#     ax_diag.text(
#         fig_w_mm / 2,
#         -5,
#         f"Figure: {fig_w_mm:.1f} mm\n({info['figure_size_px'][0]} px @ {dpi} DPI)",
#         ha="center",
#         va="top",
#         fontsize=10,
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
#     )
# 
#     ax_diag.annotate(
#         "",
#         xy=(-3, fig_h_mm),
#         xytext=(-3, 0),
#         arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
#     )
#     ax_diag.text(
#         -5,
#         fig_h_mm / 2,
#         f"Figure: {fig_h_mm:.1f} mm\n({info['figure_size_px'][1]} px)",
#         ha="right",
#         va="center",
#         fontsize=10,
#         rotation=90,
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
#     )
# 
#     # Axes dimensions
#     axes_center_x = margin_left_mm + axes_w_mm / 2
#     axes_center_y = margin_bottom_mm + axes_h_mm / 2
# 
#     ax_diag.annotate(
#         "",
#         xy=(margin_left_mm + axes_w_mm, axes_center_y),
#         xytext=(margin_left_mm, axes_center_y),
#         arrowprops=dict(arrowstyle="<->", color="blue", lw=2),
#     )
#     ax_diag.text(
#         axes_center_x,
#         axes_center_y + 1,
#         f"Axes: {axes_w_mm:.1f} mm\n({info['axes_size_px'][0]} px)",
#         ha="center",
#         va="bottom",
#         fontsize=11,
#         fontweight="bold",
#         color="blue",
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
#     )
# 
#     ax_diag.annotate(
#         "",
#         xy=(axes_center_x, margin_bottom_mm + axes_h_mm),
#         xytext=(axes_center_x, margin_bottom_mm),
#         arrowprops=dict(arrowstyle="<->", color="blue", lw=2),
#     )
#     ax_diag.text(
#         axes_center_x + 1,
#         axes_center_y,
#         f"Axes: {axes_h_mm:.1f} mm\n({info['axes_size_px'][1]} px)",
#         ha="left",
#         va="center",
#         fontsize=11,
#         fontweight="bold",
#         color="blue",
#         rotation=90,
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
#     )
# 
#     # Margin annotations
#     # Left margin
#     if margin_left_mm > 1:
#         ax_diag.text(
#             margin_left_mm / 2,
#             margin_bottom_mm + axes_h_mm / 2,
#             f"L: {margin_left_mm:.1f}",
#             ha="center",
#             va="center",
#             fontsize=9,
#             color="gray",
#         )
# 
#     # Right margin
#     if margin_right_mm > 1:
#         ax_diag.text(
#             margin_left_mm + axes_w_mm + margin_right_mm / 2,
#             margin_bottom_mm + axes_h_mm / 2,
#             f"R: {margin_right_mm:.1f}",
#             ha="center",
#             va="center",
#             fontsize=9,
#             color="gray",
#         )
# 
#     # Bottom margin
#     if margin_bottom_mm > 1:
#         ax_diag.text(
#             margin_left_mm + axes_w_mm / 2,
#             margin_bottom_mm / 2,
#             f"B: {margin_bottom_mm:.1f}",
#             ha="center",
#             va="center",
#             fontsize=9,
#             color="gray",
#         )
# 
#     # Top margin
#     if margin_top_mm > 1:
#         ax_diag.text(
#             margin_left_mm + axes_w_mm / 2,
#             margin_bottom_mm + axes_h_mm + margin_top_mm / 2,
#             f"T: {margin_top_mm:.1f}",
#             ha="center",
#             va="center",
#             fontsize=9,
#             color="gray",
#         )
# 
#     # Rulers
#     if show_rulers:
#         # Horizontal ruler (bottom)
#         ruler_y = -10
#         for mm in range(0, int(fig_w_mm) + 1, 5):
#             ax_diag.plot([mm, mm], [ruler_y, ruler_y - 1], "k-", lw=0.5)
#             if mm % 10 == 0:
#                 ax_diag.plot([mm, mm], [ruler_y, ruler_y - 2], "k-", lw=1)
#                 ax_diag.text(mm, ruler_y - 3, f"{mm}", ha="center", fontsize=8)
# 
#         # Vertical ruler (left)
#         ruler_x = -10
#         for mm in range(0, int(fig_h_mm) + 1, 5):
#             ax_diag.plot([ruler_x, ruler_x - 1], [mm, mm], "k-", lw=0.5)
#             if mm % 10 == 0:
#                 ax_diag.plot([ruler_x, ruler_x - 2], [mm, mm], "k-", lw=1)
#                 ax_diag.text(ruler_x - 3, mm, f"{mm}", ha="right", fontsize=8)
# 
#     # Grid
#     if show_grid:
#         # 5mm grid
#         for mm in range(0, int(fig_w_mm) + 1, 5):
#             ax_diag.axvline(mm, color="lightgray", lw=0.5, alpha=0.5)
#         for mm in range(0, int(fig_h_mm) + 1, 5):
#             ax_diag.axhline(mm, color="lightgray", lw=0.5, alpha=0.5)
# 
#     # Set limits and aspect
#     margin_extra = 15
#     ax_diag.set_xlim(-margin_extra, fig_w_mm + 5)
#     ax_diag.set_ylim(-margin_extra, fig_h_mm + 5)
#     ax_diag.set_aspect("equal")
# 
#     # Labels
#     ax_diag.set_xlabel("Width (mm)", fontsize=12)
#     ax_diag.set_ylabel("Height (mm)", fontsize=12)
# 
#     # Title
#     title = f"Dimension Viewer\n"
#     title += f"Figure: {fig_w_mm:.1f}×{fig_h_mm:.1f} mm ({info['figure_size_px'][0]}×{info['figure_size_px'][1]} px)\n"
#     title += f"Axes: {axes_w_mm:.1f}×{axes_h_mm:.1f} mm ({info['axes_size_px'][0]}×{info['axes_size_px'][1]} px) @ {dpi} DPI"
#     ax_diag.set_title(title, fontsize=12, fontweight="bold", pad=20)
# 
#     # Legend
#     ax_diag.legend(loc="upper right", fontsize=10)
# 
#     plt.tight_layout()
# 
#     # Save if path provided
#     if output_path:
#         fig_diag.savefig(output_path, dpi=150, bbox_inches="tight")
#         print(f"✅ Dimension viewer saved to: {output_path}")
# 
#     return fig_diag
# 
# 
# def compare_modes(axes_width_mm=30, axes_height_mm=21, output_path=None):
#     """
#     Compare publication vs display modes side-by-side.
# 
#     Creates a comparison figure showing the same axes in both modes.
# 
#     Parameters
#     ----------
#     axes_width_mm : float, optional
#         Axes width in mm (default: 30)
#     axes_height_mm : float, optional
#         Axes height in mm (default: 21)
#     output_path : str, optional
#         If provided, save comparison to this path
# 
#     Returns
#     -------
#     fig_comp : matplotlib.figure.Figure
#         Comparison figure
# 
#     Examples
#     --------
#     >>> from scitex.plt.utils import compare_modes
#     >>> fig = compare_modes(30, 21, output_path='/tmp/mode_comparison.png')
#     >>> plt.show()
#     """
#     from ._figure_from_axes_mm import (
#         create_axes_with_size_mm,
#         print_dimension_info,
#     )
# 
#     # Create publication mode
#     print("\n📐 PUBLICATION MODE:")
#     print("-" * 60)
#     fig_pub, ax_pub = create_axes_with_size_mm(
#         axes_width_mm=axes_width_mm,
#         axes_height_mm=axes_height_mm,
#         mode="publication",
#     )
#     x = np.linspace(0, 2 * np.pi, 100)
#     ax_pub.plot(x, np.sin(x), "b-", lw=1)
#     ax_pub.set_title("Publication Mode", fontsize=8)
#     ax_pub.set_xlabel("X", fontsize=7)
#     ax_pub.set_ylabel("Y", fontsize=7)
#     print_dimension_info(fig_pub, ax_pub)
# 
#     # Create display mode
#     print("\n🖥️  DISPLAY MODE:")
#     print("-" * 60)
#     fig_disp, ax_disp = create_axes_with_size_mm(
#         axes_width_mm=axes_width_mm,
#         axes_height_mm=axes_height_mm,
#         mode="display",
#     )
#     ax_disp.plot(x, np.sin(x), "b-", lw=1)
#     ax_disp.set_title("Display Mode (3x scaled)", fontsize=8)
#     ax_disp.set_xlabel("X", fontsize=7)
#     ax_disp.set_ylabel("Y", fontsize=7)
#     print_dimension_info(fig_disp, ax_disp)
# 
#     # Create comparison figure
#     fig_comp = plt.figure(figsize=(12, 5))
# 
#     # Publication mode view
#     ax1 = fig_comp.add_subplot(121)
#     view_pub = view_dimensions(fig_pub, ax_pub, show_rulers=False)
#     ax1.set_title(
#         f"Publication Mode\n{axes_width_mm}×{axes_height_mm} mm @ 300 DPI",
#         fontweight="bold",
#     )
# 
#     # Display mode view
#     ax2 = fig_comp.add_subplot(122)
#     view_disp = view_dimensions(fig_disp, ax_disp, show_rulers=False)
#     ax2.set_title(
#         f"Display Mode\n{axes_width_mm * 3}×{axes_height_mm * 3} mm @ 100 DPI (3x scaled)",
#         fontweight="bold",
#     )
# 
#     plt.suptitle("Mode Comparison: Publication vs Display", fontsize=14, y=0.98)
#     plt.tight_layout()
# 
#     # Close original figures
#     plt.close(fig_pub)
#     plt.close(fig_disp)
#     plt.close(view_pub)
#     plt.close(view_disp)
# 
#     # Save if path provided
#     if output_path:
#         fig_comp.savefig(output_path, dpi=150, bbox_inches="tight")
#         print(f"\n✅ Mode comparison saved to: {output_path}")
# 
#     return fig_comp
# 
# 
# if __name__ == "__main__":
#     from ._figure_from_axes_mm import create_axes_with_size_mm
# 
#     print("=" * 60)
#     print("DIMENSION VIEWER DEMO")
#     print("=" * 60)
# 
#     # Create a figure
#     print("\n1. Creating a 30×21 mm axes (publication mode)")
#     fig, ax = create_axes_with_size_mm(
#         axes_width_mm=30,
#         axes_height_mm=21,
#         mode="publication",
#         style_mm={
#             "axis_thickness_mm": 0.2,
#             "tick_length_mm": 0.8,
#         },
#     )
# 
#     # Plot something
#     x = np.linspace(0, 2 * np.pi, 100)
#     ax.plot(x, np.sin(x), "b-", lw=1)
#     ax.set_xlabel("X axis")
#     ax.set_ylabel("Y axis")
#     ax.set_title("Sample Plot")
# 
#     # View dimensions
#     print("\n2. Viewing dimensions")
#     fig_diag = view_dimensions(fig, ax, output_path="/tmp/dimension_viewer_demo.png")
# 
#     # Compare modes
#     print("\n3. Comparing publication vs display modes")
#     fig_comp = compare_modes(30, 21, output_path="/tmp/mode_comparison.png")
# 
#     print("\n✅ Demo complete!")
#     print("Check /tmp/dimension_viewer_demo.png and /tmp/mode_comparison.png")
# 
#     plt.show()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_dimension_viewer.py
# --------------------------------------------------------------------------------
