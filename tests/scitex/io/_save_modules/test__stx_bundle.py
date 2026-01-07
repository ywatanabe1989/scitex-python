# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_stx_bundle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-19
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_stx_bundle.py
#
# """Save functions for FTS bundle format (.zip or directory)."""
#
# from pathlib import Path
#
#
# def save_stx_bundle(obj, spath, as_zip=True, bundle_type=None, basename=None, **kwargs):
#     """Save an object as an FTS bundle (.zip or directory).
#
#     FTS (Figure-Table-Statistics) is the unified bundle format that supports:
#     - figure: Publication figures with multiple panels
#     - plot: Single matplotlib plots
#     - stats: Statistical results
#
#     The content type is auto-detected from the object:
#     - FTS instance -> delegates to FTS.save()
#     - matplotlib.figure.Figure -> plot
#     - dict with 'panels' or 'elements' -> figure
#     - dict with 'comparisons' -> stats
#
#     Bundle structure:
#         output/                 # or output.zip
#             node.json           # Bundle metadata
#             encoding.json       # Data-to-visual mappings
#             theme.json          # Visual styling
#             data/               # Raw data files
#             exports/            # PNG, SVG, PDF exports
#
#     Parameters
#     ----------
#     obj : Any
#         Object to save (FTS, Figure, dict, etc.)
#     spath : str or Path
#         Output path (e.g., "output.zip" or "output/")
#     as_zip : bool
#         If True (default), save as ZIP archive. Use False for directory.
#     bundle_type : str, optional
#         Force bundle type: 'figure', 'plot', or 'stats'. Auto-detected if None.
#     **kwargs
#         Additional arguments passed to format-specific savers.
#     """
#     from scitex.io.bundle import FTS
#
#     if isinstance(obj, FTS):
#         # Delegate to FTS.save()
#         obj.save()
#         return
#
#     p = Path(spath)
#
#     # Extract basename from path if not provided
#     if basename is None:
#         basename = p.stem
#
#     # Auto-detect content type from object
#     content_type = bundle_type
#     if content_type is None:
#         import matplotlib.figure
#
#         if isinstance(obj, matplotlib.figure.Figure):
#             content_type = "plot"
#         elif hasattr(obj, "figure"):
#             content_type = "plot"
#             obj = obj.figure
#         elif isinstance(obj, dict):
#             if "panels" in obj or "elements" in obj:
#                 content_type = "figure"
#             elif "comparisons" in obj:
#                 content_type = "stats"
#             else:
#                 content_type = "figure"  # Default for dicts
#         else:
#             raise ValueError(
#                 f"Cannot auto-detect bundle type for {type(obj).__name__}. "
#                 "Please specify bundle_type='figure', 'plot', or 'stats'."
#             )
#
#     # Route to appropriate handler based on content type
#     if content_type == "plot":
#         from ._plot_stx import save_plot_as_stx
#
#         save_plot_as_stx(obj, spath, as_zip=as_zip, basename=basename, **kwargs)
#     elif content_type == "figure":
#         from scitex.io.bundle import FTS
#
#         bundle = FTS(spath, create=True, node_type="figure")
#         if isinstance(obj, dict):
#             if "title" in obj:
#                 bundle.node.title = obj["title"]
#             if "description" in obj:
#                 bundle.node.description = obj["description"]
#         bundle.save()
#     elif content_type == "stats":
#         import scitex.stats as sstats
#
#         sstats.save_statsz(obj, spath, as_zip=as_zip, **kwargs)
#     else:
#         raise ValueError(f"Unknown bundle type: {content_type}")
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_stx_bundle.py
# --------------------------------------------------------------------------------
