"""Tests for SVG metadata reading."""

import json
import os
import tempfile

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")

from scitex.io._metadata_modules.read_metadata_svg import read_metadata_svg


class TestReadMetadataSvg:
    """Tests for read_metadata_svg function."""

    def test_read_existing_metadata(self):
        """Test reading metadata from SVG with embedded metadata."""
        svg_content = '''<?xml version="1.0"?>
<svg xmlns:scitex="http://scitex.io/metadata" xmlns="http://www.w3.org/2000/svg">
<metadata id="scitex_metadata"><scitex:data>{"test": "value", "num": 42}</scitex:data></metadata>
<rect width="100" height="100"/>
</svg>'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False, encoding="utf-8"
        ) as f:
            f.write(svg_content)
            svg_path = f.name

        try:
            result = read_metadata_svg(svg_path)
            assert result == {"test": "value", "num": 42}
        finally:
            os.unlink(svg_path)

    def test_read_no_metadata_returns_none(self):
        """Test that SVG without metadata returns None."""
        svg_content = '''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
<rect width="100" height="100"/>
</svg>'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False, encoding="utf-8"
        ) as f:
            f.write(svg_content)
            svg_path = f.name

        try:
            result = read_metadata_svg(svg_path)
            assert result is None
        finally:
            os.unlink(svg_path)

    def test_read_invalid_json_returns_raw(self):
        """Test that invalid JSON is returned as raw."""
        svg_content = '''<?xml version="1.0"?>
<svg xmlns:scitex="http://scitex.io/metadata" xmlns="http://www.w3.org/2000/svg">
<metadata id="scitex_metadata"><scitex:data>not valid json</scitex:data></metadata>
</svg>'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False, encoding="utf-8"
        ) as f:
            f.write(svg_content)
            svg_path = f.name

        try:
            result = read_metadata_svg(svg_path)
            assert result == {"raw": "not valid json"}
        finally:
            os.unlink(svg_path)

    def test_read_multiline_metadata(self):
        """Test reading metadata that spans multiple lines."""
        metadata = {"key1": "value1", "key2": "value2", "nested": {"a": 1}}
        svg_content = f'''<?xml version="1.0"?>
<svg xmlns:scitex="http://scitex.io/metadata" xmlns="http://www.w3.org/2000/svg">
<metadata id="scitex_metadata"><scitex:data>{json.dumps(metadata, indent=2)}</scitex:data></metadata>
</svg>'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False, encoding="utf-8"
        ) as f:
            f.write(svg_content)
            svg_path = f.name

        try:
            result = read_metadata_svg(svg_path)
            assert result == metadata
        finally:
            os.unlink(svg_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_svg.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_svg.py
# 
# """SVG metadata reading from <metadata> element."""
# 
# import json
# import re
# from typing import Any, Dict, Optional
# 
# 
# def read_metadata_svg(image_path: str) -> Optional[Dict[str, Any]]:
#     """
#     Read metadata from an SVG file.
# 
#     Args:
#         image_path: Path to the SVG file.
# 
#     Returns:
#         Dictionary containing metadata, or None if no metadata found.
#     """
#     metadata = None
# 
#     with open(image_path, "r", encoding="utf-8") as f:
#         svg_content = f.read()
# 
#     # Look for scitex metadata element
#     match = re.search(
#         r'<metadata[^>]*id="scitex_metadata"[^>]*>.*?'
#         r"<scitex:data>(.*?)</scitex:data>.*?</metadata>",
#         svg_content,
#         flags=re.DOTALL,
#     )
#     if match:
#         metadata_json = match.group(1)
#         try:
#             metadata = json.loads(metadata_json)
#         except json.JSONDecodeError:
#             metadata = {"raw": metadata_json}
# 
#     return metadata
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_svg.py
# --------------------------------------------------------------------------------
