#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/_lines.py

"""
Line2D artist extraction.

Handles extraction of line plots, including special handling for
boxplot, violin, and stem semantic components.
"""

from typing import List, Optional, Tuple

from ._base import ExtractionContext, color_to_hex


def extract_lines(ctx: ExtractionContext) -> List[dict]:
    """Extract Line2D artists from axes."""
    from .._csv import _get_csv_column_names

    artists = []

    for i, line in enumerate(ctx.mpl_ax.lines):
        scitex_id = getattr(line, "_scitex_id", None)
        label = line.get_label()

        # Determine semantic type for special plot types
        semantic_type, semantic_id, box_idx = _get_line_semantic_info(
            ctx, i, line, scitex_id, label
        )

        # Skip internal artists for certain plot types
        if _should_skip_line(ctx, scitex_id, label, semantic_type):
            continue

        artist = _build_line_artist(
            ctx, i, line, scitex_id, label, semantic_type, semantic_id, box_idx
        )

        # Add data_ref for non-semantic lines
        if not semantic_type:
            trace_id = _get_trace_id_for_line(ctx, i, scitex_id, artist.get("id"))
            artist["data_ref"] = _get_csv_column_names(trace_id, ctx.ax_row, ctx.ax_col)
        elif ctx.is_stem and scitex_id:
            artist["data_ref"] = _get_csv_column_names(
                scitex_id, ctx.ax_row, ctx.ax_col
            )
            if semantic_type == "stem_baseline":
                artist["derived"] = True
                artist["data_ref"]["derived_from"] = "y=0"

        # Add boxplot statistics
        if (
            semantic_type == "boxplot_median"
            and box_idx is not None
            and box_idx < len(ctx.boxplot_stats)
        ):
            artist["stats"] = ctx.boxplot_stats[box_idx]

        artists.append(artist)

    return artists


def _get_line_semantic_info(
    ctx: ExtractionContext,
    index: int,
    line,
    scitex_id: Optional[str],
    label: str,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Get semantic type info for a line."""
    semantic_type = None
    semantic_id = None
    box_idx = None

    # Stem detection
    if ctx.is_stem:
        semantic_type, semantic_id = _detect_stem_semantic(line, index)

    # Boxplot detection for unlabeled lines
    if (
        ctx.skip_unlabeled
        and not scitex_id
        and label.startswith("_")
        and ctx.is_boxplot
        and ctx.num_boxes > 0
    ):
        semantic_type, semantic_id, box_idx = _detect_boxplot_semantic(
            ctx.num_boxes, index
        )
    elif (
        ctx.skip_unlabeled and not scitex_id and label.startswith("_") and ctx.is_violin
    ):
        semantic_type = "violin_component"
        semantic_id = f"violin_line_{index}"

    return semantic_type, semantic_id, box_idx


def _detect_stem_semantic(line, index: int) -> Tuple[str, str]:
    """Detect stem plot semantic type."""
    marker = line.get_marker()
    linestyle = line.get_linestyle()

    if marker and marker != "None" and linestyle == "None":
        return "stem_marker", "stem_markers"
    elif linestyle and linestyle != "None":
        ydata = line.get_ydata()
        if len(ydata) >= 2 and len(set(ydata)) == 1:
            return "stem_baseline", "stem_baseline"
        else:
            return "stem_stem", "stem_lines"
    else:
        return "stem_component", f"stem_{index}"


def _detect_boxplot_semantic(num_boxes: int, index: int) -> Tuple[str, str, int]:
    """Detect boxplot semantic type based on line index."""
    total_whiskers = 2 * num_boxes
    total_caps = 2 * num_boxes
    total_medians = num_boxes

    if index < total_whiskers:
        box_idx = index // 2
        whisker_idx = index % 2
        return "boxplot_whisker", f"box_{box_idx}_whisker_{whisker_idx}", box_idx
    elif index < total_whiskers + total_caps:
        cap_i = index - total_whiskers
        box_idx = cap_i // 2
        cap_idx = cap_i % 2
        return "boxplot_cap", f"box_{box_idx}_cap_{cap_idx}", box_idx
    elif index < total_whiskers + total_caps + total_medians:
        box_idx = index - total_whiskers - total_caps
        return "boxplot_median", f"box_{box_idx}_median", box_idx
    else:
        flier_idx = index - total_whiskers - total_caps - total_medians
        box_idx = flier_idx if flier_idx < num_boxes else num_boxes - 1
        return "boxplot_flier", f"box_{box_idx}_flier", box_idx


def _should_skip_line(
    ctx: ExtractionContext,
    scitex_id: Optional[str],
    label: str,
    semantic_type: Optional[str],
) -> bool:
    """Check if line should be skipped."""
    if ctx.skip_unlabeled and not scitex_id and label.startswith("_"):
        # Allow boxplot, violin, stem semantic types
        if ctx.is_boxplot or ctx.is_violin or ctx.is_stem:
            return False
        return True
    return False


def _build_line_artist(
    ctx: ExtractionContext,
    index: int,
    line,
    scitex_id: Optional[str],
    label: str,
    semantic_type: Optional[str],
    semantic_id: Optional[str],
    box_idx: Optional[int],
) -> dict:
    """Build artist dict for a line."""
    artist = {}

    # ID assignment
    if semantic_id and ctx.is_stem:
        artist["id"] = semantic_id
        if scitex_id:
            artist["group_id"] = scitex_id
    elif scitex_id:
        artist["id"] = scitex_id
    elif semantic_id:
        artist["id"] = semantic_id
    elif not label.startswith("_"):
        artist["id"] = label
    else:
        artist["id"] = f"line_{index}"

    # Semantic layer
    artist["mark"] = "line"
    if semantic_type:
        artist["role"] = semantic_type

    # Legend
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
        "props": {},
    }

    color_hex = color_to_hex(line.get_color())
    if color_hex:
        backend["props"]["color"] = color_hex

    backend["props"]["linestyle"] = line.get_linestyle()
    backend["props"]["linewidth_pt"] = line.get_linewidth()

    marker = line.get_marker()
    if marker and marker != "None" and marker != "none":
        backend["props"]["marker"] = marker
        backend["props"]["markersize_pt"] = line.get_markersize()
    else:
        backend["props"]["marker"] = None

    artist["backend"] = backend

    return artist


def _get_trace_id_for_line(
    ctx: ExtractionContext,
    index: int,
    scitex_id: Optional[str],
    artist_id: str,
) -> str:
    """Get trace ID for data_ref."""
    if scitex_id:
        return scitex_id

    # Try to find from history
    if hasattr(ctx.ax_for_detection, "history"):
        plot_records = []
        for record_id, record in ctx.ax_for_detection.history.items():
            if isinstance(record, tuple) and len(record) >= 2:
                if record[1] == "plot":
                    tracking_id = record[0]
                    if tracking_id.startswith("ax_"):
                        parts = tracking_id.split("_")
                        if len(parts) >= 4:
                            trace_id = "_".join(parts[3:])
                        else:
                            trace_id = parts[-1]
                    elif tracking_id.startswith("plot_"):
                        trace_id = (
                            tracking_id[5:] if len(tracking_id) > 5 else str(index)
                        )
                    else:
                        trace_id = tracking_id
                    plot_records.append(trace_id)

        if plot_records and index < len(plot_records):
            return plot_records[index]

    return artist_id


# EOF
