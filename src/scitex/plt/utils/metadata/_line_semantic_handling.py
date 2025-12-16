#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_line_semantic_handling.py

"""
Semantic handling for special line types (boxplot, violin, stem).

This module provides functions to handle semantic labeling and statistics
computation for boxplot, violin, and stem plots.
"""


def _compute_boxplot_stats(ax_for_detection):
    """
    Compute boxplot statistics from axes history.

    Parameters
    ----------
    ax_for_detection : axes wrapper
        Axes wrapper with history

    Returns
    -------
    tuple
        (num_boxes, boxplot_stats) where boxplot_stats is a list of dicts
    """
    num_boxes = 0
    boxplot_stats = []
    boxplot_data = None

    if not hasattr(ax_for_detection, "history"):
        return num_boxes, boxplot_stats

    for record in ax_for_detection.history.values():
        if isinstance(record, tuple) and len(record) >= 3:
            method_name = record[1]
            if method_name == "boxplot":
                tracked_dict = record[2]
                args = tracked_dict.get("args", [])
                if args and len(args) > 0:
                    data = args[0]
                    if hasattr(data, '__len__') and not isinstance(data, str):
                        # Check if it's list of arrays or single array
                        if hasattr(data[0], '__len__') and not isinstance(data[0], str):
                            num_boxes = len(data)
                            boxplot_data = data
                        else:
                            num_boxes = 1
                            boxplot_data = [data]
                break

    # Compute statistics
    if boxplot_data is not None:
        import numpy as np
        for box_idx, box_data in enumerate(boxplot_data):
            try:
                arr = np.asarray(box_data)
                arr = arr[~np.isnan(arr)]
                if len(arr) > 0:
                    q1 = float(np.percentile(arr, 25))
                    median = float(np.median(arr))
                    q3 = float(np.percentile(arr, 75))
                    iqr = q3 - q1
                    whisker_low = float(max(arr.min(), q1 - 1.5 * iqr))
                    whisker_high = float(min(arr.max(), q3 + 1.5 * iqr))
                    fliers = arr[(arr < whisker_low) | (arr > whisker_high)]
                    boxplot_stats.append({
                        "box_index": box_idx,
                        "median": median,
                        "q1": q1,
                        "q3": q3,
                        "whisker_low": whisker_low,
                        "whisker_high": whisker_high,
                        "n_fliers": int(len(fliers)),
                        "n_samples": int(len(arr)),
                    })
            except (ValueError, TypeError):
                pass

    return num_boxes, boxplot_stats


def _determine_semantic_type(line, i, plot_type, num_boxes, skip_unlabeled, scitex_id):
    """
    Determine semantic type and ID for a line.

    Parameters
    ----------
    line : matplotlib.lines.Line2D
        The line object
    i : int
        Line index
    plot_type : str
        Detected plot type
    num_boxes : int
        Number of boxes for boxplot
    skip_unlabeled : bool
        Whether to skip unlabeled lines
    scitex_id : str
        Scitex ID attribute

    Returns
    -------
    tuple
        (semantic_type, semantic_id, has_boxplot_stats, box_idx, should_skip)
    """
    label = line.get_label()
    semantic_type = None
    semantic_id = None
    has_boxplot_stats = False
    box_idx = None
    should_skip = False

    is_stem = plot_type == "stem"
    is_boxplot = plot_type == "boxplot"
    is_violin = plot_type == "violin"

    # For stem, always detect semantic type
    if is_stem:
        marker = line.get_marker()
        linestyle = line.get_linestyle()
        if marker and marker != "None" and linestyle == "None":
            semantic_type = "stem_marker"
            semantic_id = "stem_markers"
        elif linestyle and linestyle != "None":
            ydata = line.get_ydata()
            if len(ydata) >= 2 and len(set(ydata)) == 1:
                semantic_type = "stem_baseline"
                semantic_id = "stem_baseline"
            else:
                semantic_type = "stem_stem"
                semantic_id = "stem_lines"
        else:
            semantic_type = "stem_component"
            semantic_id = f"stem_{i}"

    if skip_unlabeled and not scitex_id and label.startswith("_"):
        # For boxplot, assign semantic roles
        if is_boxplot and num_boxes > 0:
            total_whiskers = 2 * num_boxes
            total_caps = 2 * num_boxes
            total_medians = num_boxes

            if i < total_whiskers:
                box_idx = i // 2
                whisker_idx = i % 2
                semantic_type = "boxplot_whisker"
                semantic_id = f"box_{box_idx}_whisker_{whisker_idx}"
            elif i < total_whiskers + total_caps:
                cap_i = i - total_whiskers
                box_idx = cap_i // 2
                cap_idx = cap_i % 2
                semantic_type = "boxplot_cap"
                semantic_id = f"box_{box_idx}_cap_{cap_idx}"
            elif i < total_whiskers + total_caps + total_medians:
                box_idx = i - total_whiskers - total_caps
                semantic_type = "boxplot_median"
                semantic_id = f"box_{box_idx}_median"
                has_boxplot_stats = True
            else:
                flier_idx = i - total_whiskers - total_caps - total_medians
                box_idx = flier_idx if flier_idx < num_boxes else num_boxes - 1
                semantic_type = "boxplot_flier"
                semantic_id = f"box_{box_idx}_flier"
        elif is_violin:
            semantic_type = "violin_component"
            semantic_id = f"violin_line_{i}"
        elif is_stem:
            pass  # Already handled above
        else:
            should_skip = True

    return semantic_type, semantic_id, has_boxplot_stats, box_idx, should_skip
