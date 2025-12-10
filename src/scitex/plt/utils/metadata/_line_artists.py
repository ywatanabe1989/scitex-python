#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_line_artists.py

"""
Line artist extraction utilities.

This module provides functions to extract Line2D artists and LineCollection artists
from matplotlib axes, including special handling for boxplot, violin, and stem plots.
"""

import matplotlib.colors as mcolors
from ._label_parsing import _get_csv_column_names
from ._line_semantic_handling import _compute_boxplot_stats, _determine_semantic_type


def _extract_line_artists(mpl_ax, ax_for_detection, plot_type, method, ax_row, ax_col, skip_unlabeled):
    """
    Extract Line2D artists from axes.

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Raw matplotlib axes
    ax_for_detection : axes wrapper
        Axes wrapper with history for plot type detection
    plot_type : str
        Detected plot type
    method : str
        Detected plotting method
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

    # Compute boxplot statistics if needed
    num_boxes, boxplot_stats = _compute_boxplot_stats(ax_for_detection)

    for i, line in enumerate(mpl_ax.lines):
        scitex_id = getattr(line, "_scitex_id", None)
        label = line.get_label()

        # Determine semantic type
        semantic_type, semantic_id, has_boxplot_stats, box_idx, should_skip = _determine_semantic_type(
            line, i, plot_type, num_boxes, skip_unlabeled, scitex_id
        )

        if should_skip:
            continue

        # Check if this is a regression line for scatter plots
        is_regression_line = False
        if plot_type == "scatter" and label.startswith("_"):
            xdata = line.get_xdata()
            if len(xdata) == 2:
                is_regression_line = True

        # Build artist dictionary
        artist = _build_line_artist_dict(
            line, i, scitex_id, semantic_id, semantic_type, is_regression_line,
            label, ax_row, ax_col, ax_for_detection, mpl_ax, boxplot_stats,
            has_boxplot_stats, box_idx
        )

        artists.append(artist)

    return artists


def _build_line_artist_dict(line, i, scitex_id, semantic_id, semantic_type, is_regression_line,
                             label, ax_row, ax_col, ax_for_detection, mpl_ax, boxplot_stats,
                             has_boxplot_stats, box_idx):
    """
    Build artist dictionary for a line.

    Parameters
    ----------
    line : matplotlib.lines.Line2D
        The line object
    i : int
        Line index
    scitex_id : str
        Scitex ID attribute
    semantic_id : str
        Semantic ID
    semantic_type : str
        Semantic type
    is_regression_line : bool
        Whether this is a regression line
    label : str
        Line label
    ax_row : int
        Row position in grid
    ax_col : int
        Column position in grid
    ax_for_detection : axes wrapper
        Axes wrapper with history
    mpl_ax : matplotlib.axes.Axes
        Raw matplotlib axes
    boxplot_stats : list
        Boxplot statistics
    has_boxplot_stats : bool
        Whether to add boxplot stats
    box_idx : int or None
        Box index

    Returns
    -------
    dict
        Artist dictionary
    """
    artist = {}

    # Store display id/label
    is_stem = semantic_type and semantic_type.startswith("stem")
    if semantic_id and is_stem:
        artist["id"] = semantic_id
        if scitex_id:
            artist["group_id"] = scitex_id
    elif scitex_id:
        artist["id"] = scitex_id
    elif semantic_id:
        artist["id"] = semantic_id
    elif is_regression_line:
        artist["id"] = f"regression_{i}"
    elif not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"line_{i}"

    # Semantic layer
    artist["mark"] = "line"
    if semantic_type:
        artist["role"] = semantic_type
    elif is_regression_line:
        artist["role"] = "regression_line"

    # Label (for legend)
    if not label.startswith("_"):
        artist["label"] = label
        artist["legend_included"] = True
    else:
        artist["legend_included"] = False

    artist["zorder"] = line.get_zorder()

    # Backend layer
    backend = {
        "name": "matplotlib",
        "artist_class": type(line).__name__,
        "props": {}
    }

    color = line.get_color()
    try:
        color_hex = mcolors.to_hex(color, keep_alpha=False)
        backend["props"]["color"] = color_hex
    except (ValueError, TypeError):
        backend["props"]["color"] = color

    backend["props"]["linestyle"] = line.get_linestyle()
    backend["props"]["linewidth_pt"] = line.get_linewidth()

    marker = line.get_marker()
    if marker and marker != "None" and marker != "none":
        backend["props"]["marker"] = marker
        backend["props"]["markersize_pt"] = line.get_markersize()
    else:
        backend["props"]["marker"] = None

    artist["backend"] = backend

    # data_ref - CSV column mapping
    if not semantic_type:
        trace_id_for_ref = _find_trace_id_for_line(scitex_id, i, mpl_ax, ax_for_detection)
        if not trace_id_for_ref:
            trace_id_for_ref = artist.get("id", str(i))
        artist["data_ref"] = _get_csv_column_names(trace_id_for_ref, ax_row, ax_col)
    elif is_stem and scitex_id:
        artist["data_ref"] = _get_csv_column_names(scitex_id, ax_row, ax_col)
        if semantic_type == "stem_baseline":
            artist["derived"] = True
            artist["data_ref"]["derived_from"] = "y=0"

    # Add boxplot statistics to median artist
    if has_boxplot_stats and box_idx is not None and box_idx < len(boxplot_stats):
        artist["stats"] = boxplot_stats[box_idx]

    return artist


def _extract_line_collection_artists(mpl_ax, ax_for_detection, plot_type, method, ax_row, ax_col):
    """
    Extract LineCollection artists (errorbar lines, stem lines, etc.).

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Raw matplotlib axes
    ax_for_detection : axes wrapper
        Axes wrapper with history
    plot_type : str
        Detected plot type
    method : str
        Detected plotting method
    ax_row : int
        Row position in grid
    ax_col : int
        Column position in grid

    Returns
    -------
    list
        List of artist dictionaries
    """
    artists = []

    for i, coll in enumerate(mpl_ax.collections):
        coll_type = type(coll).__name__

        if coll_type != "LineCollection":
            continue

        artist = {}

        scitex_id = getattr(coll, "_scitex_id", None)
        label = coll.get_label() if hasattr(coll, "get_label") else ""

        if scitex_id:
            artist["id"] = scitex_id
        elif label and not label.startswith("_"):
            artist["id"] = label
        else:
            artist["id"] = f"linecollection_{i}"

        # Semantic layer
        artist["mark"] = "line"
        if plot_type == "bar" or method == "barh":
            artist["role"] = "errorbar"
        elif plot_type == "stem":
            artist["role"] = "stem_stem"
            artist["id"] = "stem_lines"
        else:
            artist["role"] = "line_collection"

        artist["legend_included"] = False
        artist["zorder"] = coll.get_zorder()

        # Backend layer
        backend = {
            "name": "matplotlib",
            "artist_class": coll_type,
            "props": {}
        }

        try:
            colors = coll.get_colors()
            if len(colors) > 0:
                backend["props"]["color"] = mcolors.to_hex(colors[0], keep_alpha=False)
        except (ValueError, TypeError, IndexError):
            pass

        try:
            linewidths = coll.get_linewidths()
            if len(linewidths) > 0:
                backend["props"]["linewidth_pt"] = float(linewidths[0])
        except (ValueError, TypeError, IndexError):
            pass

        artist["backend"] = backend

        # Add data_ref
        _add_linecollection_data_ref(artist, ax_for_detection, method, ax_row, ax_col)

        artists.append(artist)

    return artists


def _add_linecollection_data_ref(artist, ax_for_detection, method, ax_row, ax_col):
    """Add data_ref to LineCollection artist."""
    if artist["role"] == "errorbar":
        errorbar_trace_id = None
        error_var = "yerr" if method == "bar" else "xerr"
        if hasattr(ax_for_detection, "history"):
            for record in ax_for_detection.history.values():
                if isinstance(record, tuple) and len(record) >= 2:
                    method_name = record[1]
                    if method_name in ("bar", "barh"):
                        errorbar_trace_id = record[0]
                        break
        if errorbar_trace_id:
            base_ref = _get_csv_column_names(errorbar_trace_id, ax_row, ax_col)
            artist["data_ref"] = {
                "x": base_ref.get("x"),
                "y": base_ref.get("y"),
                error_var: f"ax-row-{ax_row}-col-{ax_col}_trace-id-{errorbar_trace_id}_variable-{error_var}"
            }
    elif artist["role"] == "stem_stem" and hasattr(ax_for_detection, "history"):
        for record in ax_for_detection.history.values():
            if isinstance(record, tuple) and len(record) >= 2:
                method_name = record[1]
                if method_name == "stem":
                    stem_trace_id = record[0]
                    artist["data_ref"] = _get_csv_column_names(stem_trace_id, ax_row, ax_col)
                    break


def _find_trace_id_for_line(scitex_id, line_index, mpl_ax, ax_for_detection):
    """
    Find the correct trace_id for a line's data_ref.

    Priority: 1) _scitex_id, 2) History record trace_id, 3) None
    """
    if scitex_id:
        return scitex_id

    if not hasattr(ax_for_detection, "history"):
        return None

    plot_records = []
    for record_id, record in ax_for_detection.history.items():
        if isinstance(record, tuple) and len(record) >= 2:
            if record[1] == "plot":
                tracking_id = record[0]
                trace_id_for_ref = None
                if tracking_id.startswith("ax_"):
                    parts = tracking_id.split("_")
                    if len(parts) >= 4:
                        trace_id_for_ref = "_".join(parts[3:])
                    elif len(parts) == 4:
                        trace_id_for_ref = parts[3]
                elif tracking_id.startswith("plot_"):
                    trace_id_for_ref = tracking_id[5:] if len(tracking_id) > 5 else str(line_index)
                else:
                    trace_id_for_ref = tracking_id

                if trace_id_for_ref:
                    plot_records.append(trace_id_for_ref)

    if not plot_records:
        return None

    # Find the index of this line among all non-semantic lines
    non_semantic_line_idx = 0
    for j, l in enumerate(mpl_ax.lines[:line_index]):
        l_label = l.get_label()
        l_scitex_id = getattr(l, "_scitex_id", None)
        l_semantic_id = getattr(l, "_scitex_semantic_id", None)
        if not l_semantic_id and not l_label.startswith("_"):
            non_semantic_line_idx += 1
        elif l_scitex_id:
            non_semantic_line_idx += 1

    if non_semantic_line_idx < len(plot_records):
        return plot_records[non_semantic_line_idx]

    return None
