# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/_read.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/_read.py
# 
# """Main read_metadata dispatcher."""
# 
# import os
# from typing import Any, Dict, Optional
# 
# 
# def read_metadata(image_path: str) -> Optional[Dict[str, Any]]:
#     """
#     Read metadata from an image or PDF file.
# 
#     Args:
#         image_path: Path to the file (PNG, JPEG, SVG, or PDF)
# 
#     Returns:
#         Dictionary containing metadata, or None if no metadata found
# 
#     Raises:
#         FileNotFoundError: If file doesn't exist
#         ValueError: If file format is not supported
# 
#     Example:
#         >>> metadata = read_metadata('result.png')
#         >>> print(metadata['experiment'])
#         'seizure_prediction_001'
#         >>> metadata = read_metadata('result.pdf')
#     """
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"File not found: {image_path}")
# 
#     path_lower = image_path.lower()
# 
#     # Dispatch to format-specific handlers
#     if path_lower.endswith(".png"):
#         from .read_metadata_png import read_metadata_png
# 
#         return read_metadata_png(image_path)
# 
#     elif path_lower.endswith((".jpg", ".jpeg")):
#         from .read_metadata_jpeg import read_metadata_jpeg
# 
#         return read_metadata_jpeg(image_path)
# 
#     elif path_lower.endswith(".svg"):
#         from .read_metadata_svg import read_metadata_svg
# 
#         return read_metadata_svg(image_path)
# 
#     elif path_lower.endswith(".pdf"):
#         from .read_metadata_pdf import read_metadata_pdf
# 
#         return read_metadata_pdf(image_path)
# 
#     else:
#         raise ValueError(
#             f"Unsupported file format: {image_path}. "
#             "Only PNG, JPEG, SVG, and PDF formats are supported."
#         )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/_read.py
# --------------------------------------------------------------------------------
