# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_utils/_get_template.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_utils/_get_template.py
# 
# """Figure template getter functions for publications."""
# 
# from typing import Any, Dict
# 
# from ._const_sizes import (
#     DEFAULT_MARGIN_MM,
#     DEFAULT_SPACING_MM,
#     NATURE_DOUBLE_COLUMN_MM,
#     NATURE_SINGLE_COLUMN_MM,
#     SCIENCE_SINGLE_COLUMN_MM,
# )
# 
# 
# def get_nature_single_column(
#     height_mm: float = 89, nrows: int = 1, ncols: int = 1
# ) -> Dict[str, Any]:
#     """Get Nature single column figure template."""
#     return {
#         "width_mm": NATURE_SINGLE_COLUMN_MM,
#         "height_mm": height_mm,
#         "nrows": nrows,
#         "ncols": ncols,
#         "dpi": 300,
#         "left_mm": DEFAULT_MARGIN_MM,
#         "right_mm": DEFAULT_MARGIN_MM,
#         "top_mm": DEFAULT_MARGIN_MM,
#         "bottom_mm": DEFAULT_MARGIN_MM,
#         "wspace_mm": DEFAULT_SPACING_MM,
#         "hspace_mm": DEFAULT_SPACING_MM,
#         "metadata": {"template": "nature_single_column", "journal": "Nature"},
#     }
# 
# 
# def get_nature_double_column(
#     height_mm: float = 120, nrows: int = 1, ncols: int = 1
# ) -> Dict[str, Any]:
#     """Get Nature double column figure template."""
#     return {
#         "width_mm": NATURE_DOUBLE_COLUMN_MM,
#         "height_mm": height_mm,
#         "nrows": nrows,
#         "ncols": ncols,
#         "dpi": 300,
#         "left_mm": DEFAULT_MARGIN_MM,
#         "right_mm": DEFAULT_MARGIN_MM,
#         "top_mm": DEFAULT_MARGIN_MM,
#         "bottom_mm": DEFAULT_MARGIN_MM,
#         "wspace_mm": DEFAULT_SPACING_MM,
#         "hspace_mm": DEFAULT_SPACING_MM,
#         "metadata": {"template": "nature_double_column", "journal": "Nature"},
#     }
# 
# 
# def get_science_single_column(
#     height_mm: float = 84, nrows: int = 1, ncols: int = 1
# ) -> Dict[str, Any]:
#     """Get Science single column figure template."""
#     return {
#         "width_mm": SCIENCE_SINGLE_COLUMN_MM,
#         "height_mm": height_mm,
#         "nrows": nrows,
#         "ncols": ncols,
#         "dpi": 300,
#         "left_mm": DEFAULT_MARGIN_MM,
#         "right_mm": DEFAULT_MARGIN_MM,
#         "top_mm": DEFAULT_MARGIN_MM,
#         "bottom_mm": DEFAULT_MARGIN_MM,
#         "wspace_mm": DEFAULT_SPACING_MM,
#         "hspace_mm": DEFAULT_SPACING_MM,
#         "metadata": {"template": "science_single_column", "journal": "Science"},
#     }
# 
# 
# def get_a4_figure(
#     width_mm: float = 180, height_mm: float = 120, nrows: int = 1, ncols: int = 1
# ) -> Dict[str, Any]:
#     """Get A4-sized figure template."""
#     return {
#         "width_mm": width_mm,
#         "height_mm": height_mm,
#         "nrows": nrows,
#         "ncols": ncols,
#         "dpi": 300,
#         "left_mm": DEFAULT_MARGIN_MM,
#         "right_mm": DEFAULT_MARGIN_MM,
#         "top_mm": DEFAULT_MARGIN_MM,
#         "bottom_mm": DEFAULT_MARGIN_MM,
#         "wspace_mm": DEFAULT_SPACING_MM,
#         "hspace_mm": DEFAULT_SPACING_MM,
#         "metadata": {"template": "a4_figure"},
#     }
# 
# 
# def get_square_figure(
#     size_mm: float = 120, nrows: int = 1, ncols: int = 1
# ) -> Dict[str, Any]:
#     """Get square figure template."""
#     return {
#         "width_mm": size_mm,
#         "height_mm": size_mm,
#         "nrows": nrows,
#         "ncols": ncols,
#         "dpi": 300,
#         "left_mm": DEFAULT_MARGIN_MM,
#         "right_mm": DEFAULT_MARGIN_MM,
#         "top_mm": DEFAULT_MARGIN_MM,
#         "bottom_mm": DEFAULT_MARGIN_MM,
#         "wspace_mm": DEFAULT_SPACING_MM,
#         "hspace_mm": DEFAULT_SPACING_MM,
#         "metadata": {"template": "square_figure"},
#     }
# 
# 
# def get_presentation_slide(
#     aspect_ratio: str = "16:9", width_mm: float = 254
# ) -> Dict[str, Any]:
#     """Get presentation slide figure template."""
#     if aspect_ratio == "16:9":
#         height_mm = width_mm * 9 / 16
#     elif aspect_ratio == "4:3":
#         height_mm = width_mm * 3 / 4
#     else:
#         raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
# 
#     return {
#         "width_mm": width_mm,
#         "height_mm": height_mm,
#         "nrows": 1,
#         "ncols": 1,
#         "dpi": 150,
#         "left_mm": DEFAULT_MARGIN_MM,
#         "right_mm": DEFAULT_MARGIN_MM,
#         "top_mm": DEFAULT_MARGIN_MM,
#         "bottom_mm": DEFAULT_MARGIN_MM,
#         "metadata": {"template": "presentation_slide", "aspect_ratio": aspect_ratio},
#     }
# 
# 
# # Template registry
# TEMPLATES = {
#     "nature_single": get_nature_single_column,
#     "nature_double": get_nature_double_column,
#     "science_single": get_science_single_column,
#     "a4": get_a4_figure,
#     "square": get_square_figure,
#     "presentation": get_presentation_slide,
# }
# 
# 
# def get_template(name: str, **kwargs) -> Dict[str, Any]:
#     """Get a figure template by name."""
#     if name not in TEMPLATES:
#         raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
#     return TEMPLATES[name](**kwargs)
# 
# 
# def list_templates() -> list:
#     """List available template names."""
#     return list(TEMPLATES.keys())
# 
# 
# __all__ = [
#     "get_nature_single_column",
#     "get_nature_double_column",
#     "get_science_single_column",
#     "get_a4_figure",
#     "get_square_figure",
#     "get_presentation_slide",
#     "get_template",
#     "list_templates",
#     "TEMPLATES",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_utils/_get_template.py
# --------------------------------------------------------------------------------
