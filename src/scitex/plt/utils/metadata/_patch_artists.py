#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_patch_artists.py

"""
Patch artist extraction utilities.

This module provides functions to extract Rectangle (bar/hist) and Wedge (pie) patches
from matplotlib axes.
"""

import matplotlib.colors as mcolors
from ._label_parsing import _get_csv_column_names


def _extract_rectangle_artists(mpl_ax, ax_for_detection, plot_type, ax_row, ax_col, skip_unlabeled):
    """
    Extract Rectangle patches (bar/barh/hist charts).

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Raw matplotlib axes
    ax_for_detection : axes wrapper
        Axes wrapper with history
    plot_type : str
        Detected plot type
    ax_row : int
        Row position in grid
    ax_col : int
        Column position in grid
    skip_unlabeled : bool
        Whether to skip unlabeled internal artists

    Returns
    -------
    list
        List of artist dictionaries
    """
    artists = []

    # First, collect all rectangles
    rectangles = []
    for i, patch in enumerate(mpl_ax.patches):
        patch_type = type(patch).__name__
        if patch_type == "Rectangle":
            rectangles.append((i, patch))

    is_bar = plot_type in ("bar", "barh")
    is_hist = plot_type == "hist"

    # Get trace_id from history for data_ref
    trace_id_for_bars = None
    if hasattr(ax_for_detection, "history"):
        for record in ax_for_detection.history.values():
            if isinstance(record, tuple) and len(record) >= 2:
                method_name = record[1]
                if method_name in ("bar", "barh", "hist"):
                    trace_id_for_bars = record[0]
                    break

    bar_count = 0
    for rect_idx, (i, patch) in enumerate(rectangles):
        patch_type = type(patch).__name__

        scitex_id = getattr(patch, "_scitex_id", None)
        label = patch.get_label() if hasattr(patch, "get_label") else ""

        # For bar/hist, we want ALL rectangles even if unlabeled
        if not (is_bar or is_hist):
            if skip_unlabeled and not scitex_id and (not label or label.startswith("_")):
                continue

        artist = {}

        # Generate unique ID with index
        base_id = scitex_id or (label if label and not label.startswith("_") else trace_id_for_bars or "bar")
        artist["id"] = f"{base_id}_{bar_count}"
        artist["group_id"] = base_id

        # Semantic layer
        artist["mark"] = "bar"
        if is_hist:
            artist["role"] = "hist_bin"
        else:
            artist["role"] = "bar_body"

        # Legend inclusion - only first bar of a group
        if label and not label.startswith("_") and bar_count == 0:
            artist["label"] = label
            artist["legend_included"] = True
        else:
            artist["legend_included"] = False

        artist["zorder"] = patch.get_zorder()

        # Backend layer
        backend = {
            "name": "matplotlib",
            "artist_class": patch_type,
            "props": {}
        }

        try:
            backend["props"]["facecolor"] = mcolors.to_hex(patch.get_facecolor(), keep_alpha=False)
        except (ValueError, TypeError):
            pass
        try:
            backend["props"]["edgecolor"] = mcolors.to_hex(patch.get_edgecolor(), keep_alpha=False)
        except (ValueError, TypeError):
            pass
        try:
            backend["props"]["linewidth_pt"] = patch.get_linewidth()
        except (ValueError, TypeError):
            pass

        artist["backend"] = backend

        # Bar geometry
        try:
            artist["geometry"] = {
                "x": patch.get_x(),
                "y": patch.get_y(),
                "width": patch.get_width(),
                "height": patch.get_height(),
            }
        except (ValueError, TypeError):
            pass

        # data_ref with row_index
        if trace_id_for_bars:
            if is_hist:
                prefix = f"ax-row-{ax_row}-col-{ax_col}_trace-id-{trace_id_for_bars}_variable-"
                artist["data_ref"] = {
                    "x": f"{prefix}bin-centers",
                    "y": f"{prefix}bin-counts",
                    "row_index": bar_count,
                    "bin_index": bar_count,
                }
            else:
                artist["data_ref"] = _get_csv_column_names(trace_id_for_bars, ax_row, ax_col)
                artist["data_ref"]["row_index"] = bar_count

        bar_count += 1
        artists.append(artist)

    return artists


def _extract_wedge_artists(mpl_ax):
    """
    Extract Wedge patches (pie charts).

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Raw matplotlib axes

    Returns
    -------
    list
        List of artist dictionaries
    """
    artists = []
    wedge_count = 0

    for i, patch in enumerate(mpl_ax.patches):
        patch_type = type(patch).__name__

        if patch_type != "Wedge":
            continue

        artist = {}

        scitex_id = getattr(patch, "_scitex_id", None)
        label = patch.get_label() if hasattr(patch, "get_label") else ""

        if scitex_id:
            artist["id"] = scitex_id
        elif label and not label.startswith("_"):
            artist["id"] = label
        else:
            artist["id"] = f"wedge_{wedge_count}"
            wedge_count += 1

        artist["mark"] = "pie"
        artist["role"] = "pie_wedge"

        if label and not label.startswith("_"):
            artist["label"] = label
            artist["legend_included"] = True
        else:
            artist["legend_included"] = False

        artist["zorder"] = patch.get_zorder()

        # Backend layer
        backend = {
            "name": "matplotlib",
            "artist_class": patch_type,
            "props": {}
        }
        try:
            backend["props"]["facecolor"] = mcolors.to_hex(patch.get_facecolor(), keep_alpha=False)
        except (ValueError, TypeError):
            pass

        artist["backend"] = backend
        artists.append(artist)

    return artists
