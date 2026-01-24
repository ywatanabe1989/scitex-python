#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_legend.py

"""
Legend extraction utilities for figure metadata.

Extracts legend information including handles and artist references.
"""

from typing import Optional


def _extract_legend_info(ax) -> Optional[dict]:
    """
    Extract legend information from axes.

    Uses matplotlib terminology for legend properties.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to extract legend from

    Returns
    -------
    dict or None
        Legend info dictionary with matplotlib properties, or None if no legend
    """
    legend = ax.get_legend()
    if legend is None:
        return None

    legend_info = {
        "visible": legend.get_visible(),
        "loc": legend._loc if hasattr(legend, "_loc") else "best",
        "frameon": legend.get_frame_on() if hasattr(legend, "get_frame_on") else True,
    }

    # ncol - number of columns
    if hasattr(legend, "_ncols"):
        legend_info["ncol"] = legend._ncols
    elif hasattr(legend, "_ncol"):
        legend_info["ncol"] = legend._ncol

    # Extract legend handles with artist references
    handles = _extract_legend_handles(ax, legend)
    if handles:
        legend_info["handles"] = handles

    return legend_info


def _extract_legend_handles(ax, legend) -> list:
    """Extract legend handles with artist references."""
    handles = []
    texts = legend.get_texts()
    legend_handles = legend.legend_handles if hasattr(legend, "legend_handles") else []

    # Get the raw matplotlib axes for accessing lines to match IDs
    mpl_ax = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax

    for i, text in enumerate(texts):
        label_text = text.get_text()
        handle_entry = {"label": label_text}

        # Try to get artist_id from corresponding handle
        artist_id = None
        if i < len(legend_handles):
            handle = legend_handles[i]
            if hasattr(handle, "_scitex_id"):
                artist_id = handle._scitex_id

        # Fallback: find matching artist by label in axes artists
        if artist_id is None:
            artist_id = _find_artist_id_by_label(mpl_ax, label_text)

        if artist_id:
            handle_entry["artist_id"] = artist_id

        handles.append(handle_entry)

    return handles


def _find_artist_id_by_label(mpl_ax, label_text: str) -> Optional[str]:
    """Find artist ID by matching label in axes artists."""
    # Check lines
    for line in mpl_ax.lines:
        line_label = line.get_label()
        if line_label == label_text:
            if hasattr(line, "_scitex_id"):
                return line._scitex_id
            elif not line_label.startswith("_"):
                return line_label

    # Check collections (scatter)
    for coll in mpl_ax.collections:
        coll_label = coll.get_label() if hasattr(coll, "get_label") else ""
        if coll_label == label_text:
            if hasattr(coll, "_scitex_id"):
                return coll._scitex_id
            elif coll_label and not coll_label.startswith("_"):
                return coll_label

    # Check patches (bar/hist/pie)
    for patch in mpl_ax.patches:
        patch_label = patch.get_label() if hasattr(patch, "get_label") else ""
        if patch_label == label_text:
            if hasattr(patch, "_scitex_id"):
                return patch._scitex_id
            elif patch_label and not patch_label.startswith("_"):
                return patch_label

    # Check images (imshow)
    for img in mpl_ax.images:
        img_label = img.get_label() if hasattr(img, "get_label") else ""
        if img_label == label_text:
            if hasattr(img, "_scitex_id"):
                return img._scitex_id
            elif img_label and not img_label.startswith("_"):
                return img_label

    return None


# EOF
