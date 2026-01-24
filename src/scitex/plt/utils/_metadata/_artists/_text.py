#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/_text.py

"""
Text artist extraction.

Handles Text annotations and labels.
"""

from typing import List

from ._base import ExtractionContext, color_to_hex


def extract_text(ctx: ExtractionContext) -> List[dict]:
    """Extract text artists from axes."""
    artists = []
    text_count = 0

    for i, text_obj in enumerate(ctx.mpl_ax.texts):
        text_content = text_obj.get_text()
        if not text_content or text_content.strip() == "":
            continue

        artist = _extract_text_artist(ctx, text_count, text_obj, text_content)
        if artist:
            artists.append(artist)
            text_count += 1

    return artists


def _extract_text_artist(
    ctx: ExtractionContext, index: int, text_obj, text_content: str
) -> dict:
    """Extract Text artist."""
    artist = {}

    scitex_id = getattr(text_obj, "_scitex_id", None)

    if scitex_id:
        artist["id"] = scitex_id
    else:
        artist["id"] = f"text_{index}"

    # Semantic layer
    artist["mark"] = "text"

    # Determine role from content
    pos = text_obj.get_position()
    if any(kw in text_content.lower() for kw in ["r=", "p=", "r²=", "n="]):
        artist["role"] = "stats_annotation"
    else:
        artist["role"] = "annotation"

    artist["legend_included"] = False
    artist["zorder"] = text_obj.get_zorder()

    # Geometry
    artist["geometry"] = {
        "x": pos[0],
        "y": pos[1],
    }

    # Text content
    artist["text"] = text_content

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(text_obj).__name__,
        "props": {},
    }

    try:
        color = text_obj.get_color()
        backend["props"]["color"] = color_to_hex(color)
    except (ValueError, TypeError):
        pass

    try:
        backend["props"]["fontsize_pt"] = text_obj.get_fontsize()
    except (ValueError, TypeError):
        pass

    try:
        backend["props"]["ha"] = text_obj.get_ha()
        backend["props"]["va"] = text_obj.get_va()
    except (ValueError, TypeError):
        pass

    artist["backend"] = backend

    # Data reference for tracked text
    if scitex_id:
        artist["data_ref"] = {
            "x": f"text_{index}_x",
            "y": f"text_{index}_y",
            "content": f"text_{index}_content",
        }

    return artist


# EOF
