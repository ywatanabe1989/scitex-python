#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:55:38 (ywatanabe)"
# File: ./scitex_repo/src/scitex/io/_load_modules/_image.py

from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image


def _load_image(lpath: str, return_metadata: bool = False, **kwargs) -> Union[Any, Tuple[Any, Optional[Dict]]]:
    """
    Load image file.

    Args:
        lpath: Path to image file
        return_metadata: If True, return (image, metadata_dict) tuple
        **kwargs: Additional arguments passed to Image.open()

    Returns:
        PIL.Image object, or (PIL.Image, dict) tuple if return_metadata=True
        If no metadata found, returns (PIL.Image, None)
    """
    supported_exts = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
    if not any(lpath.lower().endswith(ext) for ext in supported_exts):
        raise ValueError("Unsupported image format")

    img = Image.open(lpath, **kwargs)

    if not return_metadata:
        return img

    # Try to read metadata
    try:
        from .._metadata import read_metadata
        metadata = read_metadata(lpath)
        return img, metadata

    except Exception as e:
        # If metadata reading fails, return None
        import warnings
        warnings.warn(f"Failed to read metadata: {e}")
        return img, None


# EOF
