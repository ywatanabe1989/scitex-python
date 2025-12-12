#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata/embed_metadata_png.py

"""PNG metadata embedding using tEXt chunks."""

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def embed_metadata_png(image_path: str, metadata_json: str) -> None:
    """
    Embed metadata into a PNG file using tEXt chunks.

    Args:
        image_path: Path to the PNG file.
        metadata_json: JSON string of metadata to embed.
    """
    img = Image.open(image_path)
    pnginfo = PngInfo()
    pnginfo.add_text("scitex_metadata", metadata_json)
    img.save(image_path, "PNG", pnginfo=pnginfo)
    img.close()


# EOF
