"""Tests for main embed_metadata dispatcher."""

import json
import os
import tempfile

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")
from PIL import Image

from scitex.io._metadata_modules import embed_metadata, read_metadata


class TestEmbedMetadataDispatcher:
    """Tests for embed_metadata dispatcher function."""

    def test_embed_png(self):
        """Test embedding metadata into PNG via dispatcher."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

        try:
            img = Image.new("RGB", (50, 50), "red")
            img.save(png_path)
            img.close()

            metadata = {"format": "png", "test": True}
            embed_metadata(png_path, metadata)

            result = read_metadata(png_path)
            assert result == metadata
        finally:
            os.unlink(png_path)

    def test_embed_svg(self):
        """Test embedding metadata into SVG via dispatcher."""
        svg_content = '''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
<rect width="100" height="100"/>
</svg>'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False, encoding="utf-8"
        ) as f:
            f.write(svg_content)
            svg_path = f.name

        try:
            metadata = {"format": "svg", "test": True}
            embed_metadata(svg_path, metadata)

            result = read_metadata(svg_path)
            assert result == metadata
        finally:
            os.unlink(svg_path)

    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            embed_metadata("/nonexistent/path.png", {"test": 1})

    def test_unsupported_format_raises_error(self):
        """Test that unsupported format raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            bmp_path = f.name

        try:
            img = Image.new("RGB", (10, 10), "red")
            img.save(bmp_path)
            img.close()

            with pytest.raises(ValueError, match="Unsupported file format"):
                embed_metadata(bmp_path, {"test": 1})
        finally:
            os.unlink(bmp_path)

    def test_case_insensitive_extension(self):
        """Test that file extension matching is case insensitive."""
        with tempfile.NamedTemporaryFile(suffix=".PNG", delete=False) as f:
            png_path = f.name

        try:
            img = Image.new("RGB", (50, 50), "blue")
            img.save(png_path)
            img.close()

            metadata = {"case": "insensitive"}
            embed_metadata(png_path, metadata)

            result = read_metadata(png_path)
            assert result == metadata
        finally:
            os.unlink(png_path)

    def test_roundtrip_all_formats(self):
        """Test metadata roundtrip for all supported formats."""
        metadata = {"roundtrip": True, "number": 123}

        # PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name
        try:
            img = Image.new("RGB", (10, 10), "white")
            img.save(png_path)
            img.close()
            embed_metadata(png_path, metadata)
            assert read_metadata(png_path) == metadata
        finally:
            os.unlink(png_path)

        # SVG
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False
        ) as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>')
            svg_path = f.name
        try:
            embed_metadata(svg_path, metadata)
            assert read_metadata(svg_path) == metadata
        finally:
            os.unlink(svg_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/_embed.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/_embed.py
# 
# """Main embed_metadata dispatcher."""
# 
# import os
# from typing import Any, Dict
# 
# from ._utils import serialize_metadata
# 
# 
# def embed_metadata(image_path: str, metadata: Dict[str, Any]) -> None:
#     """
#     Embed metadata into an existing image or PDF file.
# 
#     Args:
#         image_path: Path to the image/PDF file (PNG, JPEG, SVG, or PDF)
#         metadata: Dictionary containing metadata (must be JSON serializable)
# 
#     Raises:
#         ValueError: If file format is not supported or metadata is not JSON serializable
#         FileNotFoundError: If file doesn't exist
# 
#     Example:
#         >>> metadata = {
#         ...     'experiment': 'seizure_prediction_001',
#         ...     'session': '2024-11-14',
#         ...     'analysis': 'PAC'
#         ... }
#         >>> embed_metadata('result.png', metadata)
#         >>> embed_metadata('result.pdf', metadata)
#     """
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"File not found: {image_path}")
# 
#     # Serialize metadata to JSON
#     metadata_json = serialize_metadata(metadata)
# 
#     path_lower = image_path.lower()
# 
#     # Dispatch to format-specific handlers
#     if path_lower.endswith(".png"):
#         from .embed_metadata_png import embed_metadata_png
# 
#         embed_metadata_png(image_path, metadata_json)
# 
#     elif path_lower.endswith((".jpg", ".jpeg")):
#         from .embed_metadata_jpeg import embed_metadata_jpeg
# 
#         embed_metadata_jpeg(image_path, metadata_json)
# 
#     elif path_lower.endswith(".svg"):
#         from .embed_metadata_svg import embed_metadata_svg
# 
#         embed_metadata_svg(image_path, metadata_json)
# 
#     elif path_lower.endswith(".pdf"):
#         from .embed_metadata_pdf import embed_metadata_pdf
# 
#         embed_metadata_pdf(image_path, metadata_json, metadata)
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/_embed.py
# --------------------------------------------------------------------------------
