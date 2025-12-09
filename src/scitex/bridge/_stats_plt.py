#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/bridge/_stats_plt.py
# Time-stamp: "2024-12-09 10:00:00 (ywatanabe)"
"""
Bridge module for stats ↔ plt integration.

Provides adapters to:
- Add statistical results as annotations on matplotlib axes
- Extract stats from axes with tracked annotations
- Format statistical text for plot display

Coordinate Convention
---------------------
This module uses **axes coordinates** (0-1 normalized) for positioning
when auto-positioning is used (no explicit x, y given). This matches
matplotlib's typical annotation workflow where positions are relative
to the axes bounding box.

- (0, 0) = bottom-left of axes
- (1, 1) = top-right of axes
- Default auto-position: (0.5, 0.95) = top-center

When explicit x, y are provided, they use whatever coordinate system
the caller intends (typically data coordinates unless transform is set).
"""

from typing import Optional, Dict, Any, Union, List
import warnings

from scitex.schema import StatResult, Position


def format_stat_for_plot(
    stat_result: StatResult,
    format_style: str = "asterisk",
) -> str:
    """
    Format a StatResult for display on a plot.

    Parameters
    ----------
    stat_result : StatResult
        The statistical result to format
    format_style : str
        One of "asterisk", "compact", "detailed", "publication"

    Returns
    -------
    str
        Formatted string for plot annotation
    """
    return stat_result.format_text(format_style)


def add_stat_to_axes(
    ax,
    stat_result: StatResult,
    x: Optional[float] = None,
    y: Optional[float] = None,
    format_style: str = "asterisk",
    **kwargs,
) -> Any:
    """
    Add a statistical result annotation to a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or scitex.plt AxisWrapper
        The axes to annotate
    stat_result : StatResult
        The statistical result to display
    x : float, optional
        X position for the annotation. If None, uses stat_result.positioning
    y : float, optional
        Y position for the annotation. If None, uses stat_result.positioning
    format_style : str
        Format style for the text ("asterisk", "compact", "detailed", "publication")
    **kwargs
        Additional kwargs passed to ax.annotate() or ax.text()

    Returns
    -------
    matplotlib.text.Text or matplotlib.text.Annotation
        The created annotation object
    """
    # Get formatted text
    text = format_stat_for_plot(stat_result, format_style)

    # Determine position
    # Check if using StatResult positioning
    use_stat_positioning = False
    if x is None or y is None:
        positioning = stat_result.positioning
        if positioning and positioning.position:
            pos = positioning.position
            if x is None:
                x = pos.x
            if y is None:
                y = pos.y
            use_stat_positioning = True

    # Auto-position: top center of axes in axes coordinates
    if x is None:
        x = 0.5
    if y is None:
        y = 0.95

    # Default to axes coordinates (0-1) unless user explicitly sets transform
    # This makes positioning intuitive: (0.5, 0.9) = top center of plot
    if "transform" not in kwargs:
        kwargs["transform"] = _get_axes_transform(ax)
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("va", "top")

    # Apply styling from StatResult if available
    styling = stat_result.styling
    if styling:
        kwargs.setdefault("fontsize", styling.font_size_pt)
        kwargs.setdefault("fontfamily", styling.font_family)
        kwargs.setdefault("color", styling.color)

    # Get the actual matplotlib axes
    mpl_ax = _get_mpl_axes(ax)

    # Create the annotation
    annotation = mpl_ax.text(x, y, text, **kwargs)

    # Store reference to stat_result on the annotation for later extraction
    annotation._scitex_stat_result = stat_result

    return annotation


def extract_stats_from_axes(
    ax,
    include_non_stat: bool = False,
) -> List[StatResult]:
    """
    Extract StatResult objects from axes annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or scitex.plt AxisWrapper
        The axes to extract stats from
    include_non_stat : bool
        If True, create basic StatResult for non-stat annotations

    Returns
    -------
    List[StatResult]
        List of StatResult objects found in annotations
    """
    results = []
    mpl_ax = _get_mpl_axes(ax)

    # Check all text objects (annotations and texts)
    for text_obj in mpl_ax.texts:
        if hasattr(text_obj, "_scitex_stat_result"):
            results.append(text_obj._scitex_stat_result)
        elif include_non_stat:
            # Create a basic StatResult for non-stat text
            content = text_obj.get_text()
            if content.strip():
                # Try to detect if it's a stat-like annotation
                result = _parse_stat_annotation(content)
                if result:
                    results.append(result)

    return results


def _get_mpl_axes(ax):
    """Get the underlying matplotlib axes from wrapper or native."""
    # Handle scitex AxisWrapper
    if hasattr(ax, "_axes_mpl"):
        return ax._axes_mpl
    # Handle scitex AxesWrapper (multiple axes)
    if hasattr(ax, "_axes_scitex"):
        axes = ax._axes_scitex
        if hasattr(axes, "flat"):
            return axes.flat[0]
        return axes
    # Already matplotlib axes
    return ax


def _get_axes_transform(ax):
    """Get the axes transform for positioning."""
    mpl_ax = _get_mpl_axes(ax)
    return mpl_ax.transAxes


def _parse_stat_annotation(text: str) -> Optional[StatResult]:
    """
    Try to parse a text annotation as a statistical result.

    Parameters
    ----------
    text : str
        Text content to parse

    Returns
    -------
    Optional[StatResult]
        Parsed StatResult or None if not parseable
    """
    from scitex.schema import create_stat_result

    text = text.strip()

    # Try to detect asterisks pattern
    if text in ["*", "**", "***", "ns", "n.s."]:
        stars = text.replace("n.s.", "ns")
        # Can't determine actual stats, create placeholder
        p_value = {
            "***": 0.0001,
            "**": 0.005,
            "*": 0.03,
            "ns": 0.5,
        }.get(stars, 0.5)
        return create_stat_result(
            test_type="unknown",
            statistic_name="stat",
            statistic_value=0.0,
            p_value=p_value,
        )

    # Try to parse patterns like "r = 0.85***" or "(t = 2.5, p < 0.01)"
    import re

    # Pattern: statistic = value[stars]
    match = re.match(r"([a-zA-Z]+)\s*=\s*([\d.-]+)(\*+|ns)?", text)
    if match:
        stat_name = match.group(1)
        stat_value = float(match.group(2))
        stars = match.group(3) or "ns"
        p_value = {
            "***": 0.0001,
            "**": 0.005,
            "*": 0.03,
            "ns": 0.5,
        }.get(stars, 0.5)
        return create_stat_result(
            test_type="unknown",
            statistic_name=stat_name,
            statistic_value=stat_value,
            p_value=p_value,
        )

    return None


__all__ = [
    "add_stat_to_axes",
    "extract_stats_from_axes",
    "format_stat_for_plot",
]


# EOF
