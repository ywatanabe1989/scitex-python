"""Tests for SVG metadata embedding."""

import json
import os
import tempfile

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")

from scitex.io._metadata_modules.embed_metadata_svg import embed_metadata_svg


class TestEmbedMetadataSvg:
    """Tests for embed_metadata_svg function."""

    def _create_svg(self, path: str, content: str = None):
        """Create a simple SVG file."""
        if content is None:
            content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <circle cx="50" cy="50" r="40" fill="red"/>
</svg>'''
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_embed_simple_metadata(self):
        """Test embedding simple metadata into SVG."""
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            self._create_svg(svg_path)
            metadata = {"test": "value", "number": 42}
            embed_metadata_svg(svg_path, json.dumps(metadata))

            with open(svg_path, "r") as f:
                content = f.read()

            assert 'xmlns:scitex="http://scitex.io/metadata"' in content
            assert '<metadata id="scitex_metadata">' in content
            assert "<scitex:data>" in content
            assert '"test": "value"' in content
        finally:
            os.unlink(svg_path)

    def test_embed_overwrites_existing_metadata(self):
        """Test that new metadata replaces existing metadata."""
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            self._create_svg(svg_path)

            # Embed first metadata
            embed_metadata_svg(svg_path, json.dumps({"first": True}))

            # Embed second metadata
            embed_metadata_svg(svg_path, json.dumps({"second": True}))

            with open(svg_path, "r") as f:
                content = f.read()

            # Should only have one metadata element
            assert content.count('<metadata id="scitex_metadata">') == 1
            assert '"second": true' in content
            assert '"first": true' not in content
        finally:
            os.unlink(svg_path)

    def test_embed_preserves_namespace_if_exists(self):
        """Test that existing scitex namespace is preserved."""
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            svg_with_ns = '''<?xml version="1.0"?>
<svg xmlns:scitex="http://scitex.io/metadata" xmlns="http://www.w3.org/2000/svg">
  <rect width="100" height="100"/>
</svg>'''
            self._create_svg(svg_path, svg_with_ns)
            embed_metadata_svg(svg_path, json.dumps({"test": 1}))

            with open(svg_path, "r") as f:
                content = f.read()

            # Should only have one namespace declaration
            assert content.count('xmlns:scitex') == 1
        finally:
            os.unlink(svg_path)

    def test_invalid_svg_raises_error(self):
        """Test that invalid SVG raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name
            f.write(b"not an svg file")

        try:
            with pytest.raises(ValueError, match="Invalid SVG file"):
                embed_metadata_svg(svg_path, json.dumps({"test": 1}))
        finally:
            os.unlink(svg_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_svg.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_svg.py
# 
# """SVG metadata embedding using <metadata> element."""
# 
# import re
# 
# 
# def embed_metadata_svg(image_path: str, metadata_json: str) -> None:
#     """
#     Embed metadata into an SVG file using <metadata> element.
# 
#     Args:
#         image_path: Path to the SVG file.
#         metadata_json: JSON string of metadata to embed.
# 
#     Raises:
#         ValueError: If the SVG file is invalid.
#     """
#     with open(image_path, "r", encoding="utf-8") as f:
#         svg_content = f.read()
# 
#     # Remove existing scitex metadata if present
#     svg_content = re.sub(
#         r'<metadata[^>]*id="scitex_metadata"[^>]*>.*?</metadata>',
#         "",
#         svg_content,
#         flags=re.DOTALL,
#     )
# 
#     # Find the opening <svg> tag and insert metadata after it
#     svg_match = re.search(r"(<svg[^>]*>)", svg_content)
#     if svg_match:
#         svg_tag_end = svg_match.end()
#         # Create metadata element with scitex data
#         metadata_element = (
#             f'\n<metadata id="scitex_metadata">'
#             f"<scitex:data>{metadata_json}</scitex:data>"
#             f"</metadata>\n"
#         )
#         svg_content = (
#             svg_content[:svg_tag_end]
#             + metadata_element
#             + svg_content[svg_tag_end:]
#         )
# 
#         # Ensure scitex namespace is declared in svg tag if not present
#         if "xmlns:scitex" not in svg_content:
#             svg_content = svg_content.replace(
#                 "<svg",
#                 '<svg xmlns:scitex="http://scitex.io/metadata"',
#                 1,
#             )
# 
#         with open(image_path, "w", encoding="utf-8") as f:
#             f.write(svg_content)
#     else:
#         raise ValueError(f"Invalid SVG file: {image_path}")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_svg.py
# --------------------------------------------------------------------------------
