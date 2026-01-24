#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/_patches.py

"""
Patch artist extraction.

Handles Rectangle (bar, hist), Wedge (pie), and Polygon patches.
"""

from typing import List, Optional

from ._base import ExtractionContext, color_to_hex


def extract_patches(ctx: ExtractionContext) -> List[dict]:
    """Extract patch artists from axes."""
    artists = []

    # Collect rectangles
    rectangles = []
    for i, patch in enumerate(ctx.mpl_ax.patches):
        patch_type = type(patch).__name__
        if patch_type == "Rectangle":
            rectangles.append((i, patch))
        elif patch_type == "Wedge":
            artist = _extract_wedge(ctx, i, patch)
            if artist:
                artists.append(artist)
        elif "Poly" in patch_type:
            artist = _extract_polygon(ctx, i, patch)
            if artist:
                artists.append(artist)

    # Extract rectangles (bar/hist)
    if rectangles:
        bar_artists = _extract_rectangles(ctx, rectangles)
        artists.extend(bar_artists)

    return artists


def _extract_rectangles(ctx: ExtractionContext, rectangles: List[tuple]) -> List[dict]:
    """Extract Rectangle patches (bar/hist)."""
    from .._csv import _get_csv_column_names

    artists = []
    is_bar = ctx.plot_type in ("bar", "barh")
    is_hist = ctx.plot_type == "hist"

    # Get trace_id from history
    trace_id_for_bars = _get_bar_trace_id(ctx)

    bar_count = 0
    for rect_idx, (i, patch) in enumerate(rectangles):
        scitex_id = getattr(patch, "_scitex_id", None)
        label = patch.get_label() if hasattr(patch, "get_label") else ""

        # Skip internal patches for non-bar/hist types
        if not (is_bar or is_hist):
            if (
                ctx.skip_unlabeled
                and not scitex_id
                and (not label or label.startswith("_"))
            ):
                continue

        artist = {}

        # Generate unique ID
        base_id = scitex_id or (
            label if label and not label.startswith("_") else trace_id_for_bars or "bar"
        )
        artist["id"] = f"{base_id}_{bar_count}"
        artist["group_id"] = base_id

        # Semantic layer
        artist["mark"] = "bar"
        if is_hist:
            artist["role"] = "hist_bin"
        else:
            artist["role"] = "bar_body"

        # Legend
        if label and not label.startswith("_") and bar_count == 0:
            artist["label"] = label
            artist["legend_included"] = True
        else:
            artist["legend_included"] = False

        artist["zorder"] = patch.get_zorder()

        # Backend layer
        backend = {
            "name": "matplotlib",
            "artist_class": type(patch).__name__,
            "props": {},
        }

        try:
            backend["props"]["facecolor"] = color_to_hex(patch.get_facecolor())
        except (ValueError, TypeError):
            pass

        try:
            backend["props"]["edgecolor"] = color_to_hex(patch.get_edgecolor())
        except (ValueError, TypeError):
            pass

        try:
            backend["props"]["linewidth_pt"] = patch.get_linewidth()
        except (ValueError, TypeError):
            pass

        artist["backend"] = backend

        # Geometry
        try:
            artist["geometry"] = {
                "x": patch.get_x(),
                "y": patch.get_y(),
                "width": patch.get_width(),
                "height": patch.get_height(),
            }
        except (ValueError, TypeError, AttributeError):
            pass

        # Data reference
        if trace_id_for_bars:
            artist["data_ref"] = _get_csv_column_names(
                trace_id_for_bars, ctx.ax_row, ctx.ax_col
            )
            artist["data_ref"]["row_index"] = bar_count

        bar_count += 1
        artists.append(artist)

    return artists


def _get_bar_trace_id(ctx: ExtractionContext) -> Optional[str]:
    """Get trace ID for bar/hist from history."""
    if hasattr(ctx.ax_for_detection, "history"):
        for record in ctx.ax_for_detection.history.values():
            if isinstance(record, tuple) and len(record) >= 2:
                method_name = record[1]
                if method_name in ("bar", "barh", "hist"):
                    return record[0]
    return None


def _extract_wedge(ctx: ExtractionContext, index: int, patch) -> dict:
    """Extract Wedge (pie) patch."""
    artist = {}
    scitex_id = getattr(patch, "_scitex_id", None)
    label = patch.get_label() if hasattr(patch, "get_label") else ""

    if scitex_id:
        artist["id"] = scitex_id
    elif label and not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"wedge_{index}"

    artist["mark"] = "wedge"
    artist["role"] = "pie_slice"

    if label and not label.startswith("_"):
        artist["label"] = label
        artist["legend_included"] = True
    else:
        artist["legend_included"] = False

    artist["zorder"] = patch.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(patch).__name__,
        "props": {},
    }

    try:
        backend["props"]["facecolor"] = color_to_hex(patch.get_facecolor())
    except (ValueError, TypeError):
        pass

    try:
        backend["props"]["edgecolor"] = color_to_hex(patch.get_edgecolor())
    except (ValueError, TypeError):
        pass

    artist["backend"] = backend

    # Geometry
    try:
        artist["geometry"] = {
            "theta1": patch.theta1,
            "theta2": patch.theta2,
            "r": patch.r,
        }
    except (ValueError, TypeError, AttributeError):
        pass

    return artist


def _extract_polygon(ctx: ExtractionContext, index: int, patch) -> dict:
    """Extract Polygon patch."""
    artist = {}
    scitex_id = getattr(patch, "_scitex_id", None)
    label = patch.get_label() if hasattr(patch, "get_label") else ""

    if scitex_id:
        artist["id"] = scitex_id
    elif label and not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"polygon_{index}"

    artist["mark"] = "polygon"
    artist["legend_included"] = False
    artist["zorder"] = patch.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(patch).__name__,
        "props": {},
    }

    try:
        backend["props"]["facecolor"] = color_to_hex(patch.get_facecolor())
    except (ValueError, TypeError):
        pass

    try:
        backend["props"]["edgecolor"] = color_to_hex(patch.get_edgecolor())
    except (ValueError, TypeError):
        pass

    artist["backend"] = backend

    return artist


# EOF
