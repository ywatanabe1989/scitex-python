# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_figure_exporter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/_latex/_figure_exporter.py
# 
# """Export FTS figures to LaTeX using includegraphics."""
# 
# from dataclasses import dataclass
# from pathlib import Path
# from typing import TYPE_CHECKING, Optional
# 
# from ._utils import escape_latex_minimal, sanitize_label
# 
# if TYPE_CHECKING:
#     from scitex.io.bundle import FTS
# 
# 
# @dataclass
# class FigureExportOptions:
#     """Options for figure LaTeX export.
# 
#     Attributes:
#         width_mm: Figure width in mm (from node.size_mm if not specified)
#         placement: Float placement specifier (htbp, t, b, h, p, H)
#         centering: Whether to center the figure
#         include_caption: Whether to include caption
#         include_label: Whether to include label
#         pdf_path: Override path to PDF file
#         relative_path: Whether to use relative paths
#     """
# 
#     width_mm: Optional[float] = None
#     placement: str = "htbp"
#     centering: bool = True
#     include_caption: bool = True
#     include_label: bool = True
#     pdf_path: Optional[str] = None
#     relative_path: bool = True
# 
# 
# def export_figure_to_latex(
#     bundle: "FTS",
#     options: Optional[FigureExportOptions] = None,
# ) -> str:
#     """Export a figure/plot bundle to LaTeX.
# 
#     Generates a figure environment with includegraphics pointing to
#     a pre-rendered PDF file.
# 
#     Args:
#         bundle: FTS bundle of type 'figure' or 'plot'
#         options: Export options
# 
#     Returns:
#         LaTeX code string
#     """
#     if options is None:
#         options = FigureExportOptions()
# 
#     node = bundle.node
#     node_type = node.type
# 
#     # Validate type
#     if node_type not in ("figure", "plot", "image"):
#         raise ValueError(f"Cannot export node type '{node_type}' as figure. Expected 'figure', 'plot', or 'image'.")
# 
#     # Get width
#     width_mm = options.width_mm
#     if width_mm is None and node.size_mm:
#         width_mm = node.size_mm.width
#     if width_mm is None:
#         width_mm = 80  # Default width
# 
#     # Get PDF path
#     if options.pdf_path:
#         pdf_path = options.pdf_path
#     else:
#         # Look for existing PDF export
#         pdf_path = _find_pdf_export(bundle, options.relative_path)
# 
#     # Get caption
#     caption = node.name or ""
# 
#     # Get label
#     label = sanitize_label(node.id)
#     label_prefix = "fig" if node_type in ("figure", "plot") else "img"
# 
#     # Build LaTeX
#     lines = []
#     lines.append(f"\\begin{{figure}}[{options.placement}]")
# 
#     if options.centering:
#         lines.append("    \\centering")
# 
#     lines.append(f"    \\includegraphics[width={width_mm}mm]{{{pdf_path}}}")
# 
#     if options.include_caption and caption:
#         escaped_caption = escape_latex_minimal(caption)
#         lines.append(f"    \\caption{{{escaped_caption}}}")
# 
#     if options.include_label:
#         lines.append(f"    \\label{{{label_prefix}:{label}}}")
# 
#     lines.append("\\end{figure}")
# 
#     return "\n".join(lines)
# 
# 
# def _find_pdf_export(bundle: "FTS", relative: bool = True) -> str:
#     """Find the PDF export path for a bundle.
# 
#     Args:
#         bundle: FTS bundle
#         relative: Return relative path
# 
#     Returns:
#         Path string for includegraphics
#     """
#     # Check exports directory
#     exports_dir = bundle.path / "exports"
#     if exports_dir.exists():
#         # Look for PDF files
#         pdf_files = list(exports_dir.glob("*.pdf"))
#         if pdf_files:
#             pdf_path = pdf_files[0]
#             if relative:
#                 # Return path relative to exports/
#                 return f"exports/{pdf_path.name}"
#             return str(pdf_path)
# 
#     # No PDF found - generate a placeholder path
#     bundle_name = bundle.path.stem
#     return f"exports/{bundle_name}.pdf"
# 
# 
# def generate_figure_preamble() -> str:
#     """Generate required LaTeX preamble for figures.
# 
#     Returns:
#         Preamble code with required packages
#     """
#     return """% Required packages for figure export
# \\usepackage{graphicx}
# \\usepackage{float}  % For [H] placement specifier
# """
# 
# 
# __all__ = [
#     "FigureExportOptions",
#     "export_figure_to_latex",
#     "generate_figure_preamble",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_figure_exporter.py
# --------------------------------------------------------------------------------
