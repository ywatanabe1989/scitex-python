# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_bundle_resolver.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/editor/edit/bundle_resolver.py
#
# """Bundle path resolution for FTS bundles (unified format)."""
#
# import json as json_module
# import tempfile
# import zipfile
# from pathlib import Path
# from typing import Tuple
#
# __all__ = [
#     "resolve_plot_bundle",
#     "resolve_figure_bundle",
#     "resolve_layered_plot_bundle",
#     "resolve_stx_bundle",
# ]
#
#
# def resolve_figure_bundle(path: Path, panel_index: int = 0) -> Tuple:
#     """
#     Resolve paths from a .figure bundle (multi-panel figure).
#
#     Uses in-memory zip reading for .plot panels - no disk extraction.
#
#     Parameters
#     ----------
#     path : Path
#         Path to .figure bundle (.figure or .figure)
#     panel_index : int, optional
#         Index of panel to open (default: 0 for first panel)
#
#     Returns
#     -------
#     tuple
#         (json_path, csv_path, png_path, hitmap_path, bundle_spec, panel_info)
#     """
#     spath = str(path)
#     figure_is_zip = False
#
#     # Handle ZIP vs directory for figz
#     if spath.endswith(".figure") and not spath.endswith(".figure"):
#         figure_is_zip = True
#         if not path.exists():
#             raise FileNotFoundError(f"Figz bundle not found: {path}")
#         # For figz zip, extract to access nested pltz
#         temp_dir = tempfile.mkdtemp(prefix="scitex_edit_figure_")
#         with zipfile.ZipFile(path, "r") as zf:
#             zf.extractall(temp_dir)
#         bundle_dir = Path(temp_dir)
#         for item in bundle_dir.iterdir():
#             if item.is_dir() and str(item).endswith(".figure"):
#                 bundle_dir = item
#                 break
#     else:
#         bundle_dir = Path(path)
#         if not bundle_dir.exists():
#             raise FileNotFoundError(f"Figz bundle directory not found: {bundle_dir}")
#
#     # Find nested pltz bundles
#     panel_paths = []
#     panel_is_zip = []
#
#     for item in sorted(bundle_dir.iterdir(), key=lambda x: x.name):
#         if item.is_dir() and str(item).endswith(".plot"):
#             panel_paths.append(str(item))
#             panel_is_zip.append(False)
#         elif item.is_file() and str(item).endswith(".plot"):
#             panel_paths.append(str(item))
#             panel_is_zip.append(True)
#
#     if not panel_paths:
#         raise FileNotFoundError(f"No .plot panels found in figz bundle: {bundle_dir}")
#
#     # Validate panel index
#     if panel_index < 0 or panel_index >= len(panel_paths):
#         panel_index = 0
#
#     selected_panel_path = panel_paths[panel_index]
#     panel_name = Path(selected_panel_path).name
#     print(f"Opening panel: {panel_name}")
#     if len(panel_paths) > 1:
#         print(f"  (Figz contains {len(panel_paths)} panels)")
#
#     # Build panel info
#     panel_names = [Path(p).name for p in panel_paths]
#     panel_info = {
#         "panels": panel_names,
#         "panel_paths": panel_paths,
#         "panel_is_zip": panel_is_zip,
#         "current_index": panel_index,
#         "figure_dir": str(bundle_dir),
#         "figure_is_zip": figure_is_zip,
#         "bundle_path": (
#             str(path) if figure_is_zip else None
#         ),  # Original figz zip path for export/download
#     }
#
#     # Resolve the selected panel
#     result = resolve_plot_bundle(Path(selected_panel_path))
#     return result + (panel_info,)
#
#
# def resolve_plot_bundle(path: Path) -> Tuple:
#     """
#     Resolve paths from a .plot bundle (directory or ZIP).
#
#     Supports both:
#     - Legacy format (single {basename}.json)
#     - Layered format v2.0 (spec.json + style.json + cache/)
#
#     Parameters
#     ----------
#     path : Path
#         Path to .plot bundle (.plot or .plot)
#
#     Returns
#     -------
#     tuple
#         (json_path, csv_path, png_path, hitmap_path, bundle_spec)
#     """
#     spath = str(path)
#
#     # Handle ZIP vs directory
#     if spath.endswith(".plot") and not spath.endswith(".plot"):
#         if not path.exists():
#             raise FileNotFoundError(f"Bundle not found: {path}")
#         temp_dir = tempfile.mkdtemp(prefix="scitex_edit_")
#         with zipfile.ZipFile(path, "r") as zf:
#             zf.extractall(temp_dir)
#         bundle_dir = Path(temp_dir)
#         for item in bundle_dir.iterdir():
#             if item.is_dir() and str(item).endswith(".plot"):
#                 bundle_dir = item
#                 break
#     else:
#         bundle_dir = Path(path)
#         if not bundle_dir.exists():
#             raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
#
#     # Check if this is a layered bundle (v2.0)
#     spec_path = bundle_dir / "spec.json"
#     if spec_path.exists():
#         return resolve_layered_plot_bundle(bundle_dir)
#
#     # === Legacy format ===
#     json_path = None
#     csv_path = None
#     png_path = None
#     svg_path = None
#     hitmap_path = None
#     bundle_spec = None
#
#     for f in bundle_dir.iterdir():
#         name = f.name
#         if name.endswith(".json") and not name.endswith(".manual.json"):
#             json_path = f
#         elif name.endswith(".csv"):
#             csv_path = f
#         elif name.endswith("_hitmap.png"):
#             hitmap_path = f
#         elif name.endswith(".svg") and "_hitmap" not in name:
#             svg_path = f
#         elif (
#             name.endswith(".png") and "_hitmap" not in name and "_overview" not in name
#         ):
#             png_path = f
#
#     # Prefer SVG for display
#     if svg_path:
#         png_path = svg_path
#
#     if json_path and json_path.exists():
#         with open(json_path) as f:
#             bundle_spec = json_module.load(f)
#
#     return (
#         json_path,
#         csv_path if csv_path and csv_path.exists() else None,
#         png_path if png_path and png_path.exists() else None,
#         hitmap_path if hitmap_path and hitmap_path.exists() else None,
#         bundle_spec,
#     )
#
#
# def resolve_layered_plot_bundle(bundle_dir: Path) -> Tuple:
#     """
#     Resolve paths from a layered FTS bundle.
#
#     FTS format structure:
#         bundle.zip (or bundle.d/)
#             canonical/
#                 spec.json       # Main specification
#                 data.csv        # Source data
#                 encoding.json   # Data-to-visual mappings
#                 theme.json      # Visual aesthetics
#             artifacts/
#                 exports/        # PNG, SVG, PDF
#                 cache/          # geometry_px.json
#
#     Parameters
#     ----------
#     bundle_dir : Path
#         Path to bundle directory.
#
#     Returns
#     -------
#     tuple
#         (json_path, csv_path, png_path, hitmap_path, bundle_spec)
#     """
#     # Load FTS bundle directly
#     bundle_data = {}
#     spec_path = bundle_dir / "canonical" / "spec.json"
#     if not spec_path.exists():
#         spec_path = bundle_dir / "spec.json"  # Legacy location
#
#     if spec_path.exists():
#         with open(spec_path) as f:
#             bundle_data["spec"] = json_module.load(f)
#
#     # Try to load encoding and theme
#     for fname in ["encoding.json", "theme.json"]:
#         fpath = bundle_dir / "canonical" / fname
#         if not fpath.exists():
#             fpath = bundle_dir / fname
#         if fpath.exists():
#             with open(fpath) as f:
#                 bundle_data[fname.replace(".json", "")] = json_module.load(f)
#
#     csv_path = None
#     png_path = None
#     hitmap_path = None
#
#     # Find CSV (check canonical/ first, then root)
#     for search_dir in [bundle_dir / "canonical", bundle_dir]:
#         for f in search_dir.glob("*.csv"):
#             csv_path = f
#             break
#         if csv_path:
#             break
#
#     # Find exports (check artifacts/exports first, then exports/)
#     for exports_dir in [bundle_dir / "artifacts" / "exports", bundle_dir / "exports"]:
#         if exports_dir.exists():
#             for f in exports_dir.iterdir():
#                 name = f.name
#                 if name.endswith("_hitmap.png"):
#                     hitmap_path = f
#                 elif name.endswith(".svg") and "_hitmap" not in name:
#                     png_path = f
#                 elif (
#                     name.endswith(".png") and "_hitmap" not in name and png_path is None
#                 ):
#                     png_path = f
#
#     # Merge spec with encoding and theme
#     bundle_spec = bundle_data.get("spec", {})
#     if "encoding" in bundle_data:
#         bundle_spec["encoding"] = bundle_data["encoding"]
#     if "theme" in bundle_data:
#         bundle_spec["theme"] = bundle_data["theme"]
#
#     if hitmap_path and "hit_regions" in bundle_spec:
#         bundle_spec["hit_regions"]["hit_map"] = str(hitmap_path.name)
#
#     return (
#         spec_path,
#         csv_path if csv_path and csv_path.exists() else None,
#         png_path if png_path and png_path.exists() else None,
#         hitmap_path if hitmap_path and hitmap_path.exists() else None,
#         bundle_spec,
#     )
#
#
# def resolve_stx_bundle(path: Path) -> Tuple:
#     """
#     Resolve paths from a .stx bundle (unified FTS format).
#
#     Parameters
#     ----------
#     path : Path
#         Path to .stx bundle (.stx, .stx.d, .zip, or directory)
#
#     Returns
#     -------
#     tuple
#         (json_path, csv_path, png_path, hitmap_path, bundle_spec, element_info)
#     """
#     from scitex.fts import FTS
#
#     spath = str(path)
#
#     # Handle ZIP vs directory
#     if spath.endswith((".stx", ".zip")) and not spath.endswith(".stx.d"):
#         if not path.exists():
#             raise FileNotFoundError(f"Bundle not found: {path}")
#         temp_dir = tempfile.mkdtemp(prefix="scitex_edit_stx_")
#         with zipfile.ZipFile(path, "r") as zf:
#             zf.extractall(temp_dir)
#         bundle_dir = Path(temp_dir)
#         for item in bundle_dir.iterdir():
#             if item.is_dir() and str(item).endswith((".stx.d", ".d")):
#                 bundle_dir = item
#                 break
#     else:
#         bundle_dir = Path(path)
#         if not bundle_dir.exists():
#             raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
#
#     # Load using FTS class
#     fts_bundle = FTS(bundle_dir)
#
#     # Find spec path (canonical/spec.json or spec.json)
#     spec_path = bundle_dir / "canonical" / "spec.json"
#     if not spec_path.exists():
#         spec_path = bundle_dir / "spec.json"
#
#     # Find exports (artifacts/exports or exports/)
#     png_path = None
#     hitmap_path = None
#     csv_path = None
#     for exports_dir in [bundle_dir / "artifacts" / "exports", bundle_dir / "exports"]:
#         if exports_dir.exists():
#             for f in exports_dir.iterdir():
#                 name = f.name
#                 if name.endswith("_hitmap.png"):
#                     hitmap_path = f
#                 elif name.endswith(".svg") and "_hitmap" not in name:
#                     png_path = f
#                 elif (
#                     name.endswith(".png") and "_hitmap" not in name and png_path is None
#                 ):
#                     png_path = f
#
#     # Find CSV (canonical/data.csv or root)
#     for search_dir in [bundle_dir / "canonical", bundle_dir]:
#         for f in search_dir.glob("*.csv"):
#             csv_path = f
#             break
#         if csv_path:
#             break
#
#     # Build bundle spec from FTS
#     bundle_spec = fts_bundle.to_dict() if hasattr(fts_bundle, "to_dict") else {}
#
#     # Build element info for editor
#     element_info = {
#         "elements": getattr(fts_bundle, "elements", []),
#         "size_mm": fts_bundle.node.size_mm if fts_bundle.node else None,
#         "bundle_dir": str(bundle_dir),
#         "bundle_path": str(path),
#         "is_directory": bundle_dir.is_dir(),
#     }
#
#     return (
#         spec_path,
#         csv_path if csv_path and csv_path.exists() else None,
#         png_path if png_path and png_path.exists() else None,
#         hitmap_path if hitmap_path and hitmap_path.exists() else None,
#         bundle_spec,
#         element_info,
#     )
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_bundle_resolver.py
# --------------------------------------------------------------------------------
