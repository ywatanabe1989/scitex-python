#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_png.py

"""PNG metadata reading from tEXt chunks."""

import json
from typing import Any, Dict, Optional

from PIL import Image


def read_metadata_png(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata from a PNG file.

    Args:
        image_path: Path to the PNG file.

    Returns:
        Dictionary containing metadata, or None if no metadata found.
    """
    metadata = None
    img = Image.open(image_path)
    try:
        if hasattr(img, "info") and "scitex_metadata" in img.info:
            metadata_json = img.info["scitex_metadata"]
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                # Metadata exists but is not valid JSON
                metadata = {"raw": metadata_json}
    finally:
        img.close()

    return metadata


# EOF
