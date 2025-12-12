#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:55:38 (ywatanabe)"
# File: ./scitex_repo/src/scitex/io/_load_modules/_image.py

from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image

from scitex import logging

logger = logging.getLogger(__name__)


def _load_image(
    lpath: str, metadata: bool = True, verbose: bool = False, **kwargs
) -> Union[Any, Tuple[Any, Optional[Dict]]]:
    """
    Load image file.

    Args:
        lpath: Path to image file
        metadata: If True, return (image, metadata_dict) tuple. Default True.
        verbose: If True, log information about metadata loading. Default False.
        **kwargs: Additional arguments passed to Image.open()

    Returns:
        By default (metadata=True): (PIL.Image, dict) tuple
        If metadata=False: PIL.Image object only
        If no metadata found, returns (PIL.Image, None)
    """
    supported_exts = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
    if not any(lpath.lower().endswith(ext) for ext in supported_exts):
        raise ValueError("Unsupported image format")

    img = Image.open(lpath, **kwargs)

    if not metadata:
        return img

    # Try to read metadata
    if verbose:
        logger.info(f"✅ Loading image with metadata from: {lpath}")

    try:
        from .._metadata import read_metadata

        metadata_dict = read_metadata(lpath)

        if verbose:
            if metadata_dict:
                logger.info(f"  • Embedded metadata found:")
                for key, value in metadata_dict.items():
                    logger.info(f"    - {key}: {value}")
            else:
                logger.info("  • No embedded metadata found")

        return img, metadata_dict

    except Exception as e:
        # If metadata reading fails, return None
        logger.warning(f"Failed to read metadata: {e}")
        return img, None


# EOF
