#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_jpeg.py

"""JPEG metadata reading from EXIF ImageDescription field."""

import json
from typing import Any, Dict, Optional

from PIL import Image


def read_metadata_jpeg(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata from a JPEG file.

    Args:
        image_path: Path to the JPEG file.

    Returns:
        Dictionary containing metadata, or None if no metadata found.
    """
    metadata = None
    img = Image.open(image_path)
    try:
        try:
            import piexif

            # Load EXIF data
            if "exif" in img.info:
                exif_dict = piexif.load(img.info["exif"])

                # Try to read ImageDescription field
                if piexif.ImageIFD.ImageDescription in exif_dict.get("0th", {}):
                    description = exif_dict["0th"][piexif.ImageIFD.ImageDescription]

                    # Decode bytes to string
                    if isinstance(description, bytes):
                        description = description.decode("utf-8", errors="ignore")

                    # Try to parse as JSON
                    try:
                        metadata = json.loads(description)
                    except json.JSONDecodeError:
                        # If not JSON, return as raw text
                        metadata = {"raw": description}
        except ImportError:
            pass  # piexif not available, return None
        except Exception:
            pass  # EXIF data corrupted or not readable
    finally:
        img.close()

    return metadata


# EOF
