# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_editor_launcher.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_editor/_cui/_editor_launcher.py
#
# """Main edit function for launching Flask visual editor."""
#
# from pathlib import Path
# from typing import Union
#
# from ._backend_detector import detect_best_backend, print_available_backends
# from ._bundle_resolver import (
#     resolve_figure_bundle,
#     resolve_plot_bundle,
# )
# from ._path_resolver import resolve_figure_paths
#
# __all__ = ["edit"]
#
#
# def edit(
#     path: Union[str, Path],
#     apply_manual: bool = True,
#     port: int = 5050,
# ) -> None:
#     """
#     Launch interactive Flask editor for figure style/annotation editing.
#
#     Parameters
#     ----------
#     path : str or Path
#         Path to figure file. Can be:
#         - .stx or .stx.d unified element bundle
#         - .plot directory bundle (recommended for hitmap selection)
#         - .plot ZIP bundle
#         - .figure or .figure multi-panel bundle
#         - JSON file (figure.json or figure.manual.json)
#         - CSV file (figure.csv) - for data-only start
#         - PNG file (figure.png)
#     apply_manual : bool, optional
#         If True, load .manual.json overrides if exists (default: True)
#     port : int, optional
#         Port number for Flask editor. Default: 5050.
#
#     Returns
#     -------
#     None
#         Editor runs in Flask event loop. Changes saved to .manual.json.
#     """
#     path = Path(path)
#     spath = str(path)
#     parent_str = str(path.parent) if path.is_file() else ""
#
#     # Panel info for multi-panel figures
#     panel_info = None
#
#     # Resolve paths based on input type
#     json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = (
#         _resolve_paths(path, spath, parent_str)
#     )
#
#     if not json_path.exists():
#         raise FileNotFoundError(f"JSON file not found: {json_path}")
#
#     # Load data
#     import scitex as stx
#
#     metadata = bundle_spec if bundle_spec else stx.io.load(json_path)
#     csv_data = None
#     if csv_path and csv_path.exists():
#         csv_data = stx.io.load(csv_path)
#
#     # Load manual overrides if exists
#     manual_path = json_path.with_suffix(".manual.json")
#     manual_overrides = None
#     if apply_manual and manual_path.exists():
#         manual_data = stx.io.load(manual_path)
#         manual_overrides = manual_data.get("overrides", {})
#
#     # Verify Flask is available
#     detect_best_backend()
#
#     # Print status
#     print_available_backends()
#     print(f"Launching Flask editor for: {json_path}")
#
#     # Launch Flask editor
#     _launch_flask(
#         json_path=json_path,
#         metadata=metadata,
#         csv_data=csv_data,
#         png_path=png_path,
#         hitmap_path=hitmap_path,
#         manual_overrides=manual_overrides,
#         port=port,
#         panel_info=panel_info,
#     )
#
#
# def _resolve_paths(path: Path, spath: str, parent_str: str) -> tuple:
#     """Resolve paths based on input type."""
#     panel_info = None
#     hitmap_path = None
#     bundle_spec = None
#
#     # Check if this is a .stx bundle (unified element format)
#     if spath.endswith(".stx.d") or spath.endswith(".stx"):
#         from ._bundle_resolver import resolve_stx_bundle
#
#         json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = (
#             resolve_stx_bundle(path)
#         )
#     # Check if this is a .figure bundle (multi-panel figure)
#     elif spath.endswith(".figure") or spath.endswith(".figure"):
#         json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = (
#             resolve_figure_bundle(path)
#         )
#     # Check if this is a .plot bundle
#     elif (
#         spath.endswith(".plot")
#         or spath.endswith(".plot")
#         or parent_str.endswith(".plot")
#     ):
#         bundle_path = path.parent if parent_str.endswith(".plot") else path
#         json_path, csv_path, png_path, hitmap_path, bundle_spec = resolve_plot_bundle(
#             bundle_path
#         )
#     # Check if file is inside a .figure
#     elif parent_str.endswith(".figure") or (
#         path.parent.parent and str(path.parent.parent).endswith(".figure")
#     ):
#         figure_path = (
#             path.parent if parent_str.endswith(".figure") else path.parent.parent
#         )
#         json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info = (
#             resolve_figure_bundle(figure_path)
#         )
#     else:
#         # Standard file paths
#         json_path, csv_path, png_path = resolve_figure_paths(path)
#
#     return json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info
#
#
# def _launch_flask(
#     json_path,
#     metadata,
#     csv_data,
#     png_path,
#     hitmap_path,
#     manual_overrides,
#     port,
#     panel_info,
# ):
#     """Launch Flask web editor."""
#     try:
#         from .._gui._flask_editor import WebEditor
#
#         editor = WebEditor(
#             json_path=json_path,
#             metadata=metadata,
#             csv_data=csv_data,
#             png_path=png_path,
#             hitmap_path=hitmap_path,
#             manual_overrides=manual_overrides,
#             port=port,
#             panel_info=panel_info,
#         )
#         editor.run()
#     except ImportError as e:
#         raise ImportError(
#             "Flask backend requires Flask. Install with: pip install flask"
#         ) from e
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_editor_launcher.py
# --------------------------------------------------------------------------------
