#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/_extract.py

"""
Main artist extraction function.

Orchestrates extraction of all artist types from matplotlib axes.
"""

from typing import List

from ._base import create_extraction_context
from ._collections import extract_collections
from ._images import extract_images
from ._lines import extract_lines
from ._patches import extract_patches
from ._text import extract_text


def _extract_artists(ax) -> List[dict]:
    """
    Extract artist information including properties and CSV column mapping.

    Uses matplotlib terminology: each drawable element is an Artist.
    Only includes artists that were explicitly created via scitex tracking,
    not internal artists created by matplotlib functions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to extract artists from

    Returns
    -------
    list
        List of artist dictionaries with:
        - id: unique identifier
        - artist_class: matplotlib class name (Line2D, PathCollection, etc.)
        - label: legend label
        - style: color, linestyle, linewidth, etc.
        - data_ref: CSV column mapping (matches columns_actual exactly)
    """
    ctx = create_extraction_context(ax)
    artists = []

    # Extract different artist types
    artists.extend(extract_lines(ctx))
    artists.extend(extract_patches(ctx))
    artists.extend(extract_collections(ctx))
    artists.extend(extract_images(ctx))
    artists.extend(extract_text(ctx))

    return artists


# EOF
