#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_precision_sections.py

"""
Section-specific precision rounding utilities.

This module provides functions to round numeric values in specific sections
of the metadata dictionary (figure, axes, style, plot, etc.).
"""

from ._precision_config import PRECISION, _round_value, _round_list


def _round_figure_section(fig_data: dict) -> dict:
    """Round values in figure section."""
    result = {}
    for key, value in fig_data.items():
        if key == "size_mm":
            result[key] = _round_list(value, PRECISION["mm"], fixed=True)
        elif key == "size_inch":
            result[key] = _round_list(value, PRECISION["inch"], fixed=True)
        elif key == "size_px":
            result[key] = [int(v) for v in value]
        elif key == "dpi":
            result[key] = int(value)
        else:
            result[key] = value
    return result


def _round_axes_section(axes_data: dict) -> dict:
    """Round values in axes section.

    Handles both flat structure (legacy) and nested structure (ax_00, ax_01, ...).
    """
    result = {}
    for key, value in axes_data.items():
        # Check if this is a nested axes key
        if key.startswith("ax_") and isinstance(value, dict):
            result[key] = _round_single_axes_data(value)
        else:
            result[key] = _round_single_axes_data({key: value}).get(key, value)
    return result


def _round_single_axes_data(ax_data: dict) -> dict:
    """Round values for a single axes' data."""
    result = {}
    for key, value in ax_data.items():
        if key == "size_mm":
            result[key] = _round_list(value, PRECISION["mm"], fixed=True)
        elif key == "size_inch":
            result[key] = _round_list(value, PRECISION["inch"], fixed=True)
        elif key == "size_px":
            result[key] = [int(v) for v in value]
        elif key in ("position", "position_ratio", "bounds_figure_fraction"):
            result[key] = _round_list(value, PRECISION["position"], fixed=True)
        elif key == "position_in_grid":
            result[key] = [int(v) for v in value]
        elif key == "margins_mm":
            result[key] = {k: _round_value(v, PRECISION["mm"], fixed=True) for k, v in value.items()}
        elif key == "margins_inch":
            result[key] = {k: _round_value(v, PRECISION["inch"], fixed=True) for k, v in value.items()}
        elif key == "bbox_mm":
            result[key] = {k: _round_value(v, PRECISION["mm"], fixed=True) for k, v in value.items()}
        elif key == "bbox_inch":
            result[key] = {k: _round_value(v, PRECISION["inch"], fixed=True) for k, v in value.items()}
        elif key == "bbox_px":
            result[key] = {k: int(v) for k, v in value.items()}
        elif key in ("xaxis", "yaxis", "xaxis_top", "yaxis_right"):
            result[key] = _round_axis_info(value)
        elif key == "legend":
            result[key] = value
        elif key == "artists":
            result[key] = [_round_artist(a) for a in value]
        else:
            result[key] = value
    return result


def _round_axis_info(axis_data: dict) -> dict:
    """Round values in axis info dictionary."""
    axis_result = {}
    for ak, av in axis_data.items():
        if ak == "lim":
            axis_result[ak] = _round_list(av, PRECISION["lim"], fixed=True)
        elif ak == "n_ticks":
            axis_result[ak] = int(av)
        else:
            axis_result[ak] = av
    return axis_result


def _round_style_section(style_data: dict) -> dict:
    """Round values in hierarchical style section with scopes."""
    result = {}
    for scope, scope_data in style_data.items():
        if scope in ("global", "axes_default", "artist_default"):
            result[scope] = {}
            for category, category_data in scope_data.items():
                if isinstance(category_data, dict):
                    result[scope][category] = _round_style_subsection(category, category_data)
                else:
                    result[scope][category] = category_data
        elif isinstance(scope_data, dict):
            result[scope] = _round_style_subsection(scope, scope_data)
        elif isinstance(scope_data, float):
            if "_mm" in scope:
                result[scope] = _round_value(scope_data, PRECISION["mm"], fixed=True)
            elif "_pt" in scope:
                result[scope] = _round_value(scope_data, 1, fixed=True)
            else:
                result[scope] = _round_value(scope_data, 2)
        elif isinstance(scope_data, int):
            result[scope] = scope_data
        else:
            result[scope] = scope_data
    return result


def _round_style_subsection(category: str, data: dict) -> dict:
    """Round values in a style subsection based on category."""
    result = {}
    for key, value in data.items():
        if isinstance(value, float):
            if "_mm" in key or category in ("axes", "ticks", "lines", "markers"):
                result[key] = _round_value(value, PRECISION["mm"], fixed=True)
            elif "_pt" in key or category in ("fonts", "padding"):
                result[key] = _round_value(value, 1, fixed=True)
            else:
                result[key] = _round_value(value, 2)
        elif isinstance(value, int):
            result[key] = value
        else:
            result[key] = value
    return result


def _round_plot_section(plot_data: dict) -> dict:
    """Round values in plot section."""
    result = {}
    for key, value in plot_data.items():
        if key == "artists":
            result[key] = [_round_artist(a) for a in value]
        elif key == "legend":
            result[key] = value
        else:
            result[key] = value
    return result


def _round_artist(artist: dict) -> dict:
    """Round values in a single artist."""
    result = {}
    for key, value in artist.items():
        if key == "style" and isinstance(value, dict):
            result[key] = _round_style_dict(value)
        elif key == "backend" and isinstance(value, dict):
            result[key] = _round_backend_dict(value)
        elif key == "geometry" and isinstance(value, dict):
            result[key] = _round_geometry_dict(value)
        elif key == "zorder":
            result[key] = int(value) if isinstance(value, (int, float)) else value
        else:
            result[key] = value
    return result


def _round_style_dict(style: dict) -> dict:
    """Round values in legacy style dict."""
    style_result = {}
    for sk, sv in style.items():
        if sk in ("linewidth_pt", "markersize_pt"):
            style_result[sk] = _round_value(sv, PRECISION["linewidth"], fixed=True)
        else:
            style_result[sk] = sv
    return style_result


def _round_backend_dict(backend: dict) -> dict:
    """Round values in backend dict."""
    backend_result = {"name": backend.get("name", "matplotlib")}
    if "artist_class" in backend:
        backend_result["artist_class"] = backend["artist_class"]
    if "props" in backend and isinstance(backend["props"], dict):
        props_result = {}
        for pk, pv in backend["props"].items():
            if pk in ("linewidth_pt", "markersize_pt"):
                props_result[pk] = _round_value(pv, PRECISION["linewidth"], fixed=True)
            elif pk == "size":
                props_result[pk] = _round_value(pv, 1, fixed=True)
            else:
                props_result[pk] = pv
        backend_result["props"] = props_result
    return backend_result


def _round_geometry_dict(geometry: dict) -> dict:
    """Round values in geometry dict."""
    geom_result = {}
    for gk, gv in geometry.items():
        if isinstance(gv, float):
            geom_result[gk] = _round_value(gv, 4, fixed=False)
        else:
            geom_result[gk] = gv
    return geom_result


# Backward compatibility alias
_round_trace = _round_artist
