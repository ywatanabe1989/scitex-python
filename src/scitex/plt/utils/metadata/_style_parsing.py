#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_style_parsing.py

"""
Style dictionary restructuring utilities.

This module provides functions to convert flat style dictionaries into
hierarchical structures with explicit scopes (global, axes_default, artist_default).
"""

from typing import Dict


def _restructure_style(flat_style: dict) -> dict:
    """
    Restructure flat style_mm dict into hierarchical structure with explicit scopes.

    Converts:
        {"axis_thickness_mm": 0.2, "tick_length_mm": 0.8, ...}
    To:
        {
            "global": {"fonts": {...}, "padding": {...}},
            "axes_default": {"axes": {...}, "ticks": {...}},
            "artist_default": {"lines": {...}, "markers": {...}}
        }

    Style scopes:
    - global: rcParams-like settings (fonts, padding) applied to entire figure
    - axes_default: default axes appearance (can be overridden per-axes)
    - artist_default: default artist appearance (can be overridden per-artist)
    """
    result = {
        "global": {
            "fonts": {},
            "padding": {},
        },
        "axes_default": {
            "axes": {},
            "ticks": {},
        },
        "artist_default": {
            "lines": {},
            "markers": {},
        },
    }

    # Mapping from flat keys to hierarchical structure (scope, category, key)
    key_mapping = {
        # Axes-level defaults
        "axis_thickness_mm": ("axes_default", "axes", "thickness_mm"),
        "axes_thickness_mm": ("axes_default", "axes", "thickness_mm"),
        "tick_length_mm": ("axes_default", "ticks", "length_mm"),
        "tick_thickness_mm": ("axes_default", "ticks", "thickness_mm"),
        "n_ticks": ("axes_default", "ticks", "n_ticks"),
        # Artist-level defaults (Line2D, markers)
        "trace_thickness_mm": ("artist_default", "lines", "thickness_mm"),
        "line_thickness_mm": ("artist_default", "lines", "thickness_mm"),
        "marker_size_mm": ("artist_default", "markers", "size_mm"),
        "scatter_size_mm": ("artist_default", "markers", "scatter_size_mm"),
        # Global defaults (rcParams-like)
        "font_family": ("global", "fonts", "family"),
        "font_family_requested": ("global", "fonts", "family_requested"),
        "font_family_actual": ("global", "fonts", "family_actual"),
        "axis_font_size_pt": ("global", "fonts", "axis_size_pt"),
        "tick_font_size_pt": ("global", "fonts", "tick_size_pt"),
        "title_font_size_pt": ("global", "fonts", "title_size_pt"),
        "legend_font_size_pt": ("global", "fonts", "legend_size_pt"),
        "suptitle_font_size_pt": ("global", "fonts", "suptitle_size_pt"),
        "annotation_font_size_pt": ("global", "fonts", "annotation_size_pt"),
        "label_pad_pt": ("global", "padding", "label_pt"),
        "tick_pad_pt": ("global", "padding", "tick_pt"),
        "title_pad_pt": ("global", "padding", "title_pt"),
    }

    for key, value in flat_style.items():
        if key in key_mapping:
            scope, category, new_key = key_mapping[key]
            result[scope][category][new_key] = value
        else:
            # Unknown keys go to a misc section or are kept at top level
            # For now, skip unknown keys to keep structure clean
            pass

    # Remove empty categories within each scope
    for scope in list(result.keys()):
        result[scope] = {k: v for k, v in result[scope].items() if v}
        # Remove empty scopes
        if not result[scope]:
            del result[scope]

    return result


def _collect_style_metadata(fig, ax) -> dict:
    """
    Collect style metadata from figure and axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to extract style from
    ax
        Axes or AxesWrapper to extract style from

    Returns
    -------
    dict
        Hierarchical style metadata
    """
    style_metadata = {}

    # Extract style_mm from figure metadata if available
    if hasattr(fig, "_scitex_metadata") and "style_mm" in fig._scitex_metadata:
        flat_style = fig._scitex_metadata["style_mm"]
        style_metadata = _restructure_style(flat_style)

    # Extract style_mm from axes metadata if available
    elif hasattr(ax, "_scitex_metadata") and "style_mm" in ax._scitex_metadata:
        flat_style = ax._scitex_metadata["style_mm"]
        style_metadata = _restructure_style(flat_style)

    # Try to extract from wrapped axes if this is an AxisWrapper
    elif hasattr(ax, "_ax"):
        mpl_ax = ax._ax
        if hasattr(mpl_ax, "_scitex_metadata") and "style_mm" in mpl_ax._scitex_metadata:
            flat_style = mpl_ax._scitex_metadata["style_mm"]
            style_metadata = _restructure_style(flat_style)

    return style_metadata


def _extract_mode_and_method(fig, ax) -> tuple:
    """
    Extract mode (display/publication) and creation method from metadata.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to extract metadata from
    ax
        Axes or AxesWrapper to extract metadata from

    Returns
    -------
    tuple
        (mode, creation_method) where each is a string or None
    """
    mode = None
    creation_method = None

    # Try figure metadata first
    if hasattr(fig, "_scitex_metadata"):
        mode = fig._scitex_metadata.get("mode")
        creation_method = fig._scitex_metadata.get("creation_method")

    # Try axes metadata if not found in figure
    if mode is None or creation_method is None:
        if hasattr(ax, "_scitex_metadata"):
            if mode is None:
                mode = ax._scitex_metadata.get("mode")
            if creation_method is None:
                creation_method = ax._scitex_metadata.get("creation_method")

        # Try wrapped axes
        elif hasattr(ax, "_ax"):
            mpl_ax = ax._ax
            if hasattr(mpl_ax, "_scitex_metadata"):
                if mode is None:
                    mode = mpl_ax._scitex_metadata.get("mode")
                if creation_method is None:
                    creation_method = mpl_ax._scitex_metadata.get("creation_method")

    return mode, creation_method
