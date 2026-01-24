#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/_collections.py

"""
Collection artist extraction.

Handles PathCollection (scatter), PolyCollection (hexbin, violin),
QuadMesh (hist2d), and LineCollection (errorbar) extraction.
"""

from typing import List

from ._base import ExtractionContext, color_to_hex


def extract_collections(ctx: ExtractionContext) -> List[dict]:
    """Extract collection artists from axes."""
    artists = []

    for i, coll in enumerate(ctx.mpl_ax.collections):
        coll_type = type(coll).__name__

        if "PathCollection" in coll_type:
            artist = _extract_scatter(ctx, i, coll)
            if artist:
                artists.append(artist)
        elif "PolyCollection" in coll_type:
            artist = _extract_poly_collection(ctx, i, coll)
            if artist:
                artists.append(artist)
        elif "QuadMesh" in coll_type:
            artist = _extract_quadmesh(ctx, i, coll)
            if artist:
                artists.append(artist)
        elif coll_type == "LineCollection":
            artist = _extract_line_collection(ctx, i, coll)
            if artist:
                artists.append(artist)

    return artists


def _extract_scatter(ctx: ExtractionContext, index: int, coll) -> dict:
    """Extract PathCollection (scatter) artist."""
    from .._csv import _get_csv_column_names

    artist = {}
    scitex_id = getattr(coll, "_scitex_id", None)
    label = coll.get_label()

    if scitex_id:
        artist["id"] = scitex_id
    elif label and not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"scatter_{index}"

    artist["mark"] = "scatter"

    if label and not label.startswith("_"):
        artist["label"] = label
        artist["legend_included"] = True
    else:
        artist["legend_included"] = False

    artist["zorder"] = coll.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(coll).__name__,
        "props": {},
    }

    try:
        facecolors = coll.get_facecolor()
        if len(facecolors) > 0:
            backend["props"]["facecolor"] = color_to_hex(facecolors[0])
    except (ValueError, TypeError, IndexError):
        pass

    try:
        edgecolors = coll.get_edgecolor()
        if len(edgecolors) > 0:
            backend["props"]["edgecolor"] = color_to_hex(edgecolors[0])
    except (ValueError, TypeError, IndexError):
        pass

    try:
        sizes = coll.get_sizes()
        if len(sizes) > 0:
            backend["props"]["size"] = float(sizes[0])
    except (ValueError, TypeError, IndexError):
        pass

    artist["backend"] = backend

    # Data reference
    artist_id = artist.get("id", str(index))
    artist["data_ref"] = _get_csv_column_names(artist_id, ctx.ax_row, ctx.ax_col)

    return artist


def _extract_poly_collection(ctx: ExtractionContext, index: int, coll) -> dict:
    """Extract PolyCollection (hexbin, violin body) artist."""
    coll_type = type(coll).__name__

    # Check if hexbin
    if hasattr(coll, "get_array"):
        arr = coll.get_array()
        if arr is not None and len(arr) > 0 and ctx.plot_type != "violin":
            return _extract_hexbin(ctx, index, coll, arr)

    # Violin body
    if ctx.plot_type == "violin":
        return _extract_violin_body(ctx, index, coll)

    return None


def _extract_hexbin(ctx: ExtractionContext, index: int, coll, arr) -> dict:
    """Extract hexbin PolyCollection."""
    artist = {}
    scitex_id = getattr(coll, "_scitex_id", None)
    label = coll.get_label() if hasattr(coll, "get_label") else ""

    if scitex_id:
        artist["id"] = scitex_id
    elif label and not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"hexbin_{index}"

    artist["mark"] = "hexbin"
    artist["role"] = "hexbin"
    artist["legend_included"] = False
    artist["zorder"] = coll.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(coll).__name__,
        "props": {},
    }

    try:
        cmap = coll.get_cmap()
        if cmap:
            backend["props"]["cmap"] = cmap.name
    except (ValueError, TypeError, AttributeError):
        pass

    try:
        backend["props"]["vmin"] = float(coll.norm.vmin) if coll.norm else None
        backend["props"]["vmax"] = float(coll.norm.vmax) if coll.norm else None
    except (ValueError, TypeError, AttributeError):
        pass

    artist["backend"] = backend

    # Result info
    try:
        artist["result"] = {
            "n_hexagons": int(len(arr)),
            "count_range": [float(arr.min()), float(arr.max())]
            if len(arr) > 0
            else None,
            "total_count": int(arr.sum()),
        }
    except (TypeError, AttributeError, ValueError):
        pass

    return artist


def _extract_violin_body(ctx: ExtractionContext, index: int, coll) -> dict:
    """Extract violin body PolyCollection."""
    artist = {}
    scitex_id = getattr(coll, "_scitex_id", None)

    if scitex_id:
        artist["id"] = f"{scitex_id}_body_{index}"
        artist["group_id"] = scitex_id
    else:
        artist["id"] = f"violin_body_{index}"

    artist["mark"] = "polygon"
    artist["role"] = "violin_body"
    artist["legend_included"] = False
    artist["zorder"] = coll.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(coll).__name__,
        "props": {},
    }

    try:
        facecolors = coll.get_facecolor()
        if len(facecolors) > 0:
            backend["props"]["facecolor"] = color_to_hex(facecolors[0])
    except (ValueError, TypeError, IndexError):
        pass

    try:
        edgecolors = coll.get_edgecolor()
        if len(edgecolors) > 0:
            backend["props"]["edgecolor"] = color_to_hex(edgecolors[0])
    except (ValueError, TypeError, IndexError):
        pass

    artist["backend"] = backend

    return artist


def _extract_quadmesh(ctx: ExtractionContext, index: int, coll) -> dict:
    """Extract QuadMesh (hist2d) artist."""
    artist = {}
    scitex_id = getattr(coll, "_scitex_id", None)
    label = coll.get_label() if hasattr(coll, "get_label") else ""

    if scitex_id:
        artist["id"] = scitex_id
    elif label and not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"hist2d_{index}"

    artist["mark"] = "hist2d"
    artist["role"] = "hist2d"
    artist["legend_included"] = False
    artist["zorder"] = coll.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(coll).__name__,
        "props": {},
    }

    try:
        cmap = coll.get_cmap()
        if cmap:
            backend["props"]["cmap"] = cmap.name
    except (ValueError, TypeError, AttributeError):
        pass

    artist["backend"] = backend

    return artist


def _extract_line_collection(ctx: ExtractionContext, index: int, coll) -> dict:
    """Extract LineCollection (errorbar, stem) artist."""
    artist = {}
    scitex_id = getattr(coll, "_scitex_id", None)
    label = coll.get_label() if hasattr(coll, "get_label") else ""

    if scitex_id:
        artist["id"] = scitex_id
    elif label and not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"linecollection_{index}"

    artist["mark"] = "line"

    # Determine role
    if ctx.plot_type == "bar" or ctx.method == "barh":
        artist["role"] = "errorbar"
    elif ctx.plot_type == "stem":
        artist["role"] = "stem_stem"
        artist["id"] = "stem_lines"
    else:
        artist["role"] = "line_collection"

    artist["legend_included"] = False
    artist["zorder"] = coll.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(coll).__name__,
        "props": {},
    }

    try:
        colors = coll.get_colors()
        if len(colors) > 0:
            backend["props"]["color"] = color_to_hex(colors[0])
    except (ValueError, TypeError, IndexError):
        pass

    try:
        linewidths = coll.get_linewidths()
        if len(linewidths) > 0:
            backend["props"]["linewidth_pt"] = float(linewidths[0])
    except (ValueError, TypeError, IndexError):
        pass

    artist["backend"] = backend

    # Data reference for errorbar/stem
    if artist["role"] == "errorbar":
        _add_errorbar_data_ref(ctx, artist)
    elif artist["role"] == "stem_stem":
        _add_stem_data_ref(ctx, artist)

    return artist


def _add_errorbar_data_ref(ctx: ExtractionContext, artist: dict) -> None:
    """Add data_ref for errorbar LineCollection."""
    from .._csv import _get_csv_column_names

    errorbar_trace_id = None
    error_var = "yerr" if ctx.method == "bar" else "xerr"

    if hasattr(ctx.ax_for_detection, "history"):
        for record in ctx.ax_for_detection.history.values():
            if isinstance(record, tuple) and len(record) >= 2:
                method_name = record[1]
                if method_name in ("bar", "barh"):
                    errorbar_trace_id = record[0]
                    break

    if errorbar_trace_id:
        base_ref = _get_csv_column_names(errorbar_trace_id, ctx.ax_row, ctx.ax_col)
        artist["data_ref"] = {
            "x": base_ref.get("x"),
            "y": base_ref.get("y"),
            error_var: f"ax-row-{ctx.ax_row}-col-{ctx.ax_col}_trace-id-{errorbar_trace_id}_variable-{error_var}",
        }


def _add_stem_data_ref(ctx: ExtractionContext, artist: dict) -> None:
    """Add data_ref for stem LineCollection."""
    from .._csv import _get_csv_column_names

    if hasattr(ctx.ax_for_detection, "history"):
        for record in ctx.ax_for_detection.history.values():
            if isinstance(record, tuple) and len(record) >= 2:
                method_name = record[1]
                if method_name == "stem":
                    stem_trace_id = record[0]
                    artist["data_ref"] = _get_csv_column_names(
                        stem_trace_id, ctx.ax_row, ctx.ax_col
                    )
                    break


# EOF
