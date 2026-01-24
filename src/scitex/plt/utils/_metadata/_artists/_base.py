#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/_base.py

"""
Base utilities for artist extraction.

Provides common context and helper functions used across artist extractors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.colors as mcolors


@dataclass
class ExtractionContext:
    """Context for artist extraction operations."""

    ax: Any  # matplotlib axes
    mpl_ax: Any  # raw matplotlib axes
    ax_for_detection: Any  # axes for plot type detection
    ax_row: int = 0
    ax_col: int = 0
    plot_type: Optional[str] = None
    method: Optional[str] = None
    skip_unlabeled: bool = False
    id_to_history: Dict[str, tuple] = field(default_factory=dict)

    # Boxplot specific
    is_boxplot: bool = False
    num_boxes: int = 0
    boxplot_data: Optional[List] = None
    boxplot_stats: List[dict] = field(default_factory=list)

    # Violin specific
    is_violin: bool = False

    # Stem specific
    is_stem: bool = False


def create_extraction_context(ax) -> ExtractionContext:
    """Create an extraction context from axes."""
    from .._detect import _detect_plot_type

    # Get axes position for CSV column naming
    ax_row, ax_col = 0, 0
    if hasattr(ax, "_scitex_metadata") and "position_in_grid" in ax._scitex_metadata:
        pos = ax._scitex_metadata["position_in_grid"]
        ax_row, ax_col = pos[0], pos[1]

    # Get the raw matplotlib axes
    mpl_ax = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax

    # Try to find scitex wrapper for plot type detection
    ax_for_detection = ax
    if not hasattr(ax, "history") and hasattr(mpl_ax, "_scitex_wrapper"):
        ax_for_detection = mpl_ax._scitex_wrapper

    # Detect plot type
    plot_type, method = _detect_plot_type(ax_for_detection)

    # Plot types where internal artists should be hidden
    internal_plot_types = {
        "boxplot",
        "violin",
        "hist",
        "bar",
        "image",
        "heatmap",
        "kde",
        "ecdf",
        "errorbar",
        "fill",
        "stem",
        "contour",
        "pie",
        "quiver",
        "stream",
    }
    skip_unlabeled = plot_type in internal_plot_types

    # Build history map
    id_to_history = {}
    if hasattr(ax_for_detection, "history"):
        for record_id, record in ax_for_detection.history.items():
            if isinstance(record, tuple) and len(record) >= 2:
                tracking_id = record[0]
                id_to_history[tracking_id] = record

    ctx = ExtractionContext(
        ax=ax,
        mpl_ax=mpl_ax,
        ax_for_detection=ax_for_detection,
        ax_row=ax_row,
        ax_col=ax_col,
        plot_type=plot_type,
        method=method,
        skip_unlabeled=skip_unlabeled,
        id_to_history=id_to_history,
        is_boxplot=plot_type == "boxplot",
        is_violin=plot_type == "violin",
        is_stem=plot_type == "stem",
    )

    # Extract boxplot info
    if ctx.is_boxplot:
        _extract_boxplot_info(ctx)

    return ctx


def _extract_boxplot_info(ctx: ExtractionContext) -> None:
    """Extract boxplot specific information."""
    import numpy as np

    if not hasattr(ctx.ax_for_detection, "history"):
        return

    for record in ctx.ax_for_detection.history.values():
        if isinstance(record, tuple) and len(record) >= 3:
            method_name = record[1]
            if method_name == "boxplot":
                tracked_dict = record[2]
                args = tracked_dict.get("args", [])
                if args and len(args) > 0:
                    data = args[0]
                    if hasattr(data, "__len__") and not isinstance(data, str):
                        if hasattr(data[0], "__len__") and not isinstance(data[0], str):
                            ctx.num_boxes = len(data)
                            ctx.boxplot_data = data
                        else:
                            ctx.num_boxes = 1
                            ctx.boxplot_data = [data]
                break

    # Compute boxplot statistics
    if ctx.boxplot_data is not None:
        for box_idx, box_data in enumerate(ctx.boxplot_data):
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
                    ctx.boxplot_stats.append(
                        {
                            "box_index": box_idx,
                            "median": median,
                            "q1": q1,
                            "q3": q3,
                            "whisker_low": whisker_low,
                            "whisker_high": whisker_high,
                            "n_fliers": int(len(fliers)),
                            "n_samples": int(len(arr)),
                        }
                    )
            except (ValueError, TypeError):
                pass


def color_to_hex(color) -> Optional[str]:
    """Convert color to hex string."""
    try:
        return mcolors.to_hex(color, keep_alpha=False)
    except (ValueError, TypeError):
        return None


def get_artist_id(
    obj,
    index: int,
    prefix: str,
    scitex_id: Optional[str] = None,
    label: Optional[str] = None,
    semantic_id: Optional[str] = None,
) -> str:
    """Get a unique ID for an artist."""
    if scitex_id:
        return scitex_id
    if semantic_id:
        return semantic_id
    if label and not label.startswith("_"):
        return label
    return f"{prefix}_{index}"


# EOF
