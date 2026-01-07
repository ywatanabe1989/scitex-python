# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_plot_bundle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-19
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_plot_bundle.py
#
# """Save matplotlib figures as .plot bundles."""
#
# import tempfile
# import warnings
# from pathlib import Path
#
# import numpy as np
#
# from scitex import logging
#
# from ._figure_utils import get_figure_with_data
#
# logger = logging.getLogger()
#
#
# def save_plot_bundle(obj, spath, as_zip=False, data=None, layered=True, **kwargs):
#     """Save a matplotlib figure as a .plot bundle.
#
#     Bundle structure v2.0 (layered - default):
#         plot.plot/
#             spec.json           # Semantic: WHAT to plot (canonical)
#             style.json          # Appearance: HOW it looks (canonical)
#             data.csv            # Raw data (immutable)
#             exports/            # PNG, SVG, hitmap
#             cache/              # geometry_px.json, render_manifest.json
#
#     Parameters
#     ----------
#     obj : matplotlib.figure.Figure
#         The figure to save.
#     spath : str or Path
#         Output path (e.g., "plot.plot" or "plot.plot").
#     as_zip : bool
#         If True, save as ZIP archive.
#     data : pandas.DataFrame, optional
#         Data to embed in the bundle as plot.csv.
#     layered : bool
#         If True (default), use new layered format (spec/style/geometry).
#     **kwargs
#         Additional arguments passed to savefig.
#     """
#     import shutil
#
#     import matplotlib.figure
#
#     from scitex.plt.io import save_layered_plot_bundle
#
#     p = Path(spath)
#
#     # Extract basename from path
#     basename = p.stem
#     if basename.endswith(".plot"):
#         basename = basename[:-5]
#     elif basename.endswith(".d"):
#         basename = Path(basename).stem
#         if basename.endswith(".plot"):
#             basename = basename[:-5]
#
#     # Extract figure from various matplotlib object types
#     fig = obj
#     if hasattr(obj, "figure"):
#         fig = obj.figure
#     elif hasattr(obj, "fig"):
#         fig = obj.fig
#
#     if not isinstance(fig, matplotlib.figure.Figure):
#         raise TypeError(f"Expected matplotlib Figure, got {type(obj).__name__}")
#
#     dpi = kwargs.pop("dpi", 300)
#
#     # === Always use layered format ===
#     # Determine bundle directory path
#     if as_zip:
#         temp_dir = Path(tempfile.mkdtemp())
#         bundle_dir = temp_dir / f"{basename}.plot"
#         zip_path = p if not str(p).endswith(".d") else Path(str(p)[:-2])
#     else:
#         bundle_dir = p if str(p).endswith(".d") else Path(str(p) + ".d")
#         temp_dir = None
#
#     # Get CSV data from figure if not provided
#     csv_df = data
#     if csv_df is None:
#         csv_source = get_figure_with_data(obj)
#         if csv_source is not None and hasattr(csv_source, "export_as_csv"):
#             try:
#                 csv_df = csv_source.export_as_csv()
#             except Exception:
#                 pass
#
#     save_layered_plot_bundle(
#         fig=fig,
#         bundle_dir=bundle_dir,
#         basename=basename,
#         dpi=dpi,
#         csv_df=csv_df,
#     )
#
#     # Compress to ZIP if requested
#     if as_zip:
#         import zipfile
#
#         with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
#             for file_path in bundle_dir.rglob("*"):
#                 if file_path.is_file():
#                     arcname = file_path.relative_to(bundle_dir.parent)
#                     zf.write(file_path, arcname)
#         shutil.rmtree(temp_dir)
#
#
# def save_plot_bundle_legacy(obj, spath, as_zip=False, data=None, **kwargs):
#     """Legacy format - kept for reference but not actively used."""
#     import hashlib
#     import tempfile
#     from pathlib import Path
#
#     import matplotlib.figure
#
#     from scitex.io.bundle import BundleType
#     from scitex.io.bundle import save as save_bundle
#
#     p = Path(spath)
#     basename = p.stem
#     if basename.endswith(".plot"):
#         basename = basename[:-5]
#     elif basename.endswith(".d"):
#         basename = Path(basename).stem
#         if basename.endswith(".plot"):
#             basename = basename[:-5]
#
#     fig = obj
#     if hasattr(obj, "figure"):
#         fig = obj.figure
#     elif hasattr(obj, "fig"):
#         fig = obj.fig
#
#     if not isinstance(fig, matplotlib.figure.Figure):
#         raise TypeError(f"Expected matplotlib Figure, got {type(obj).__name__}")
#
#     dpi = kwargs.pop("dpi", 300)
#     fig_width_inch, fig_height_inch = fig.get_size_inches()
#
#     # Build spec
#     spec = {
#         "schema": {"name": "scitex.plt.plot", "version": "1.0.0"},
#         "backend": "mpl",
#         "data": {
#             "source": f"{basename}.csv",
#             "path": f"{basename}.csv",
#             "hash": None,
#             "columns": [],
#         },
#         "size": {
#             "width_inch": round(fig_width_inch, 2),
#             "height_inch": round(fig_height_inch, 2),
#             "width_mm": round(fig_width_inch * 25.4, 2),
#             "height_mm": round(fig_height_inch * 25.4, 2),
#             "width_px": int(fig_width_inch * dpi),
#             "height_px": int(fig_height_inch * dpi),
#             "dpi": dpi,
#             "crop_margin_mm": 1.0,
#         },
#         "axes": [],
#         "theme": {
#             "mode": "light",
#             "colors": {
#                 "background": "transparent",
#                 "axes_bg": "white",
#                 "text": "black",
#                 "spine": "black",
#                 "tick": "black",
#             },
#         },
#     }
#
#     extracted_data = {}
#
#     # Extract axes metadata
#     for i, ax in enumerate(fig.axes):
#         bbox = ax.get_position()
#         ax_info = {
#             "xlabel": ax.get_xlabel() or None,
#             "ylabel": ax.get_ylabel() or None,
#             "title": ax.get_title() or None,
#             "xlim": [round(v, 2) for v in ax.get_xlim()],
#             "ylim": [round(v, 2) for v in ax.get_ylim()],
#             "plot_type": "line",
#             "bbox": {
#                 "x0": round(bbox.x0, 4),
#                 "y0": round(bbox.y0, 4),
#                 "x1": round(bbox.x1, 4),
#                 "y1": round(bbox.y1, 4),
#                 "width": round(bbox.width, 4),
#                 "height": round(bbox.height, 4),
#             },
#             "bbox_mm": {
#                 "x0": round(bbox.x0 * fig_width_inch * 25.4, 2),
#                 "y0": round(bbox.y0 * fig_height_inch * 25.4, 2),
#                 "x1": round(bbox.x1 * fig_width_inch * 25.4, 2),
#                 "y1": round(bbox.y1 * fig_height_inch * 25.4, 2),
#                 "width": round(bbox.width * fig_width_inch * 25.4, 2),
#                 "height": round(bbox.height * fig_height_inch * 25.4, 2),
#             },
#             "bbox_px": {
#                 "x0": int(bbox.x0 * fig_width_inch * dpi),
#                 "y0": int(bbox.y0 * fig_height_inch * dpi),
#                 "x1": int(bbox.x1 * fig_width_inch * dpi),
#                 "y1": int(bbox.y1 * fig_height_inch * dpi),
#                 "width": int(bbox.width * fig_width_inch * dpi),
#                 "height": int(bbox.height * fig_height_inch * dpi),
#             },
#         }
#
#         if hasattr(ax, "_scitex_axes_width_mm"):
#             ax_info["axes_width_mm"] = ax._scitex_axes_width_mm
#         else:
#             ax_info["axes_width_mm"] = round(bbox.width * fig_width_inch * 25.4, 1)
#
#         if hasattr(ax, "_scitex_axes_height_mm"):
#             ax_info["axes_height_mm"] = ax._scitex_axes_height_mm
#         else:
#             ax_info["axes_height_mm"] = round(bbox.height * fig_height_inch * 25.4, 1)
#
#         lines_info = []
#         for j, line in enumerate(ax.get_lines()):
#             label = line.get_label()
#             if label is None or label.startswith("_"):
#                 label = f"series_{j}"
#             xdata, ydata = line.get_data()
#             if len(xdata) > 0:
#                 col_x = f"{label}_x" if i == 0 else f"ax{i}_{label}_x"
#                 col_y = f"{label}_y" if i == 0 else f"ax{i}_{label}_y"
#                 extracted_data[col_x] = np.array(xdata)
#                 extracted_data[col_y] = np.array(ydata)
#
#                 color = line.get_color()
#                 if isinstance(color, (list, tuple)):
#                     import matplotlib.colors as mcolors
#
#                     color = mcolors.to_hex(color)
#
#                 lines_info.append(
#                     {
#                         "label": label,
#                         "x_col": col_x,
#                         "y_col": col_y,
#                         "color": color,
#                         "linewidth": line.get_linewidth(),
#                     }
#                 )
#
#         if lines_info:
#             ax_info["lines"] = lines_info
#
#         spec["axes"].append(ax_info)
#
#     # Handle theme
#     if hasattr(fig, "_scitex_theme"):
#         theme_mode = fig._scitex_theme
#         spec["theme"]["mode"] = theme_mode
#         if theme_mode == "dark":
#             spec["theme"]["colors"] = {
#                 "background": "transparent",
#                 "axes_bg": "transparent",
#                 "text": "#e8e8e8",
#                 "spine": "#e8e8e8",
#                 "tick": "#e8e8e8",
#             }
#             from scitex.plt.utils._figure_mm import _apply_theme_colors
#
#             for ax in fig.axes:
#                 _apply_theme_colors(ax, theme="dark")
#
#     bundle_data = {"spec": spec, "basename": basename}
#
#     # Process data
#     csv_df = None
#     if data is not None:
#         csv_df = data
#         bundle_data["data"] = data
#     else:
#         csv_source = get_figure_with_data(obj)
#         if csv_source is not None and hasattr(csv_source, "export_as_csv"):
#             try:
#                 csv_df = csv_source.export_as_csv()
#                 if csv_df is not None and not csv_df.empty:
#                     bundle_data["data"] = csv_df
#             except Exception:
#                 csv_df = None
#
#         if csv_df is None and extracted_data:
#             try:
#                 import pandas as pd
#
#                 max_len = max(len(v) for v in extracted_data.values())
#                 padded = {}
#                 for k, v in extracted_data.items():
#                     if len(v) < max_len:
#                         padded[k] = np.pad(
#                             v, (0, max_len - len(v)), constant_values=np.nan
#                         )
#                     else:
#                         padded[k] = v
#                 csv_df = pd.DataFrame(padded)
#                 bundle_data["data"] = csv_df
#             except ImportError:
#                 pass
#
#     if csv_df is not None:
#         csv_str = csv_df.to_csv(index=False)
#         csv_hash = hashlib.sha256(csv_str.encode()).hexdigest()
#         spec["data"]["hash"] = f"sha256:{csv_hash[:16]}"
#         spec["data"]["columns"] = list(csv_df.columns)
#
#     # Save figures and hitmaps (simplified - full implementation in original)
#     crop_box = None
#     color_map = {}
#
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmp_path = Path(tmp_dir)
#
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", message=".*tight_layout.*")
#
#             png_path = tmp_path / "plot.png"
#             fig.savefig(
#                 png_path,
#                 dpi=dpi,
#                 bbox_inches="tight",
#                 format="png",
#                 transparent=True,
#             )
#
#             svg_path = tmp_path / "plot.svg"
#             fig.savefig(svg_path, bbox_inches="tight", format="svg")
#
#             pdf_path = tmp_path / "plot.pdf"
#             fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
#
#             with open(png_path, "rb") as f:
#                 bundle_data["png"] = f.read()
#
#             with open(svg_path, "rb") as f:
#                 bundle_data["svg"] = f.read()
#
#             with open(pdf_path, "rb") as f:
#                 bundle_data["pdf"] = f.read()
#
#     save_bundle(bundle_data, p, bundle_type=BundleType.PLTZ, as_zip=as_zip)
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_plot_bundle.py
# --------------------------------------------------------------------------------
