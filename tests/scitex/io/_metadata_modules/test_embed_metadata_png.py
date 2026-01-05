"""Tests for PNG metadata embedding."""

import json
import os
import tempfile

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")
from PIL import Image

from scitex.io._metadata_modules.embed_metadata_png import embed_metadata_png


class TestEmbedMetadataPng:
    """Tests for embed_metadata_png function."""

    def test_embed_simple_metadata(self):
        """Test embedding simple metadata into PNG."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

        try:
            # Create test image
            img = Image.new("RGB", (100, 100), "red")
            img.save(png_path)
            img.close()

            # Embed metadata
            metadata = {"test": "value", "number": 42}
            metadata_json = json.dumps(metadata)
            embed_metadata_png(png_path, metadata_json)

            # Verify metadata was embedded
            with Image.open(png_path) as img:
                assert "scitex_metadata" in img.info
                assert json.loads(img.info["scitex_metadata"]) == metadata
        finally:
            os.unlink(png_path)

    def test_embed_overwrites_existing_metadata(self):
        """Test that new metadata overwrites existing metadata."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

        try:
            img = Image.new("RGB", (50, 50), "blue")
            img.save(png_path)
            img.close()

            # Embed first metadata
            embed_metadata_png(png_path, json.dumps({"first": True}))

            # Embed second metadata
            embed_metadata_png(png_path, json.dumps({"second": True}))

            # Verify only second metadata exists
            with Image.open(png_path) as img:
                result = json.loads(img.info["scitex_metadata"])
                assert result == {"second": True}
        finally:
            os.unlink(png_path)

    def test_embed_unicode_metadata(self):
        """Test embedding metadata with unicode characters."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name

        try:
            img = Image.new("RGB", (50, 50), "green")
            img.save(png_path)
            img.close()

            metadata = {"title": "テスト", "emoji": "🔬"}
            embed_metadata_png(png_path, json.dumps(metadata, ensure_ascii=False))

            with Image.open(png_path) as img:
                result = json.loads(img.info["scitex_metadata"])
                assert result == metadata
        finally:
            os.unlink(png_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_png.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata/embed_metadata_png.py
# 
# """PNG metadata embedding using tEXt chunks."""
# 
# from PIL import Image
# from PIL.PngImagePlugin import PngInfo
# 
# 
# def embed_metadata_png(image_path: str, metadata_json: str) -> None:
#     """
#     Embed metadata into a PNG file using tEXt chunks.
# 
#     Args:
#         image_path: Path to the PNG file.
#         metadata_json: JSON string of metadata to embed.
#     """
#     img = Image.open(image_path)
#     pnginfo = PngInfo()
#     pnginfo.add_text("scitex_metadata", metadata_json)
#     img.save(image_path, "PNG", pnginfo=pnginfo)
#     img.close()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_png.py
# --------------------------------------------------------------------------------
