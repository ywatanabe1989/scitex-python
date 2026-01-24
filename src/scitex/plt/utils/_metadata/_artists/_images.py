#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/_images.py

"""
Image artist extraction.

Handles AxesImage (imshow, matshow) extraction.
"""

from typing import List

from ._base import ExtractionContext


def extract_images(ctx: ExtractionContext) -> List[dict]:
    """Extract image artists from axes."""
    artists = []

    for i, img in enumerate(ctx.mpl_ax.images):
        artist = _extract_image(ctx, i, img)
        if artist:
            artists.append(artist)

    return artists


def _extract_image(ctx: ExtractionContext, index: int, img) -> dict:
    """Extract AxesImage artist."""
    artist = {}
    img_type = type(img).__name__

    scitex_id = getattr(img, "_scitex_id", None)
    label = img.get_label() if hasattr(img, "get_label") else ""

    if scitex_id:
        artist["id"] = scitex_id
    elif label and not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"image_{index}"

    # Semantic layer
    artist["mark"] = "image"
    artist["role"] = "image"

    artist["legend_included"] = False
    artist["zorder"] = img.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": img_type,
        "props": {},
    }

    try:
        cmap = img.get_cmap()
        if cmap:
            backend["props"]["cmap"] = cmap.name
    except (ValueError, TypeError, AttributeError):
        pass

    try:
        backend["props"]["vmin"] = float(img.norm.vmin) if img.norm else None
        backend["props"]["vmax"] = float(img.norm.vmax) if img.norm else None
    except (ValueError, TypeError, AttributeError):
        pass

    try:
        backend["props"]["interpolation"] = img.get_interpolation()
    except (ValueError, TypeError, AttributeError):
        pass

    artist["backend"] = backend

    return artist


# EOF
