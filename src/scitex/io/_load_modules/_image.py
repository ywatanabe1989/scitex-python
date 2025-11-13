#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:55:38 (ywatanabe)"
# File: ./scitex_repo/src/scitex/io/_load_modules/_image.py

from typing import Any, Tuple, Union
import pandas as pd

from PIL import Image


def _load_image(lpath: str, return_metadata: bool = False, **kwargs) -> Union[Any, Tuple[Any, pd.DataFrame]]:
    """
    Load image file.

    Args:
        lpath: Path to image file
        return_metadata: If True, return (image, metadata_df) tuple
        **kwargs: Additional arguments passed to Image.open()

    Returns:
        PIL.Image object, or (PIL.Image, pd.DataFrame) tuple if return_metadata=True
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

        if metadata is None:
            # No metadata found, return empty DataFrame
            meta_df = pd.DataFrame()
        else:
            # Convert dict to DataFrame (single row with each key as column)
            meta_df = pd.DataFrame([metadata])

        return img, meta_df

    except Exception as e:
        # If metadata reading fails, return empty DataFrame
        import warnings
        warnings.warn(f"Failed to read metadata: {e}")
        meta_df = pd.DataFrame()
        return img, meta_df


# EOF
