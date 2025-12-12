#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/_embed.py

"""Main embed_metadata dispatcher."""

import os
from typing import Any, Dict

from ._utils import serialize_metadata


def embed_metadata(image_path: str, metadata: Dict[str, Any]) -> None:
    """
    Embed metadata into an existing image or PDF file.

    Args:
        image_path: Path to the image/PDF file (PNG, JPEG, SVG, or PDF)
        metadata: Dictionary containing metadata (must be JSON serializable)

    Raises:
        ValueError: If file format is not supported or metadata is not JSON serializable
        FileNotFoundError: If file doesn't exist

    Example:
        >>> metadata = {
        ...     'experiment': 'seizure_prediction_001',
        ...     'session': '2024-11-14',
        ...     'analysis': 'PAC'
        ... }
        >>> embed_metadata('result.png', metadata)
        >>> embed_metadata('result.pdf', metadata)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Serialize metadata to JSON
    metadata_json = serialize_metadata(metadata)

    path_lower = image_path.lower()

    # Dispatch to format-specific handlers
    if path_lower.endswith(".png"):
        from .embed_metadata_png import embed_metadata_png

        embed_metadata_png(image_path, metadata_json)

    elif path_lower.endswith((".jpg", ".jpeg")):
        from .embed_metadata_jpeg import embed_metadata_jpeg

        embed_metadata_jpeg(image_path, metadata_json)

    elif path_lower.endswith(".svg"):
        from .embed_metadata_svg import embed_metadata_svg

        embed_metadata_svg(image_path, metadata_json)

    elif path_lower.endswith(".pdf"):
        from .embed_metadata_pdf import embed_metadata_pdf

        embed_metadata_pdf(image_path, metadata_json, metadata)

    else:
        raise ValueError(
            f"Unsupported file format: {image_path}. "
            "Only PNG, JPEG, SVG, and PDF formats are supported."
        )


# EOF
