# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata.py
# 
# """
# Image and PDF metadata embedding and extraction for research reproducibility.
# 
# This module re-exports from _metadata_modules for backwards compatibility.
# See _metadata_modules/ for format-specific implementations:
# - PNG: tEXt chunks
# - JPEG: EXIF ImageDescription field
# - SVG: <metadata> element with scitex namespace
# - PDF: XMP metadata (industry standard)
# """
# 
# from ._metadata_modules import (
#     embed_metadata,
#     read_metadata,
#     has_metadata,
#     embed_metadata_png,
#     embed_metadata_jpeg,
#     embed_metadata_svg,
#     embed_metadata_pdf,
#     read_metadata_png,
#     read_metadata_jpeg,
#     read_metadata_svg,
#     read_metadata_pdf,
# )
# 
# # Backwards compatibility alias
# _convert_for_json = None  # Removed - use _metadata_modules._utils.convert_for_json
# 
# __all__ = [
#     "embed_metadata",
#     "read_metadata",
#     "has_metadata",
#     "embed_metadata_png",
#     "embed_metadata_jpeg",
#     "embed_metadata_svg",
#     "embed_metadata_pdf",
#     "read_metadata_png",
#     "read_metadata_jpeg",
#     "read_metadata_svg",
#     "read_metadata_pdf",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata.py
# --------------------------------------------------------------------------------
