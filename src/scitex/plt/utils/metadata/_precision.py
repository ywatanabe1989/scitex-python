#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_precision.py

"""
Numeric precision utilities for metadata rounding.

This module provides utilities to round numeric values in metadata dictionaries
with appropriate precision based on the type of measurement (mm, inch, px, etc.).

This module has been refactored: the implementation is now split across multiple
specialized modules. This file serves as a backward compatibility layer.
"""

from typing import Dict, List, Union

# Import from specialized modules
from ._precision_config import PRECISION, FixedFloat, _round_value, _round_list
from ._precision_sections import (
    _round_figure_section,
    _round_axes_section,
    _round_single_axes_data,
    _round_axis_info,
    _round_style_section,
    _round_style_subsection,
    _round_plot_section,
    _round_artist,
    _round_style_dict,
    _round_backend_dict,
    _round_geometry_dict,
    _round_trace,  # Backward compatibility alias
)

__all__ = [
    "PRECISION",
    "FixedFloat",
    "_round_value",
    "_round_list",
    "_round_dict",
    "_round_metadata",
    "_round_figure_section",
    "_round_axes_section",
    "_round_single_axes_data",
    "_round_axis_info",
    "_round_style_section",
    "_round_style_subsection",
    "_round_plot_section",
    "_round_artist",
    "_round_trace",  # Backward compatibility
]


def _round_dict(d: dict, precision_map: dict = None) -> dict:
    """
    Round all float values in a dict based on key-specific precision.

    Parameters
    ----------
    d : dict
        Dictionary to process
    precision_map : dict, optional
        Mapping of key patterns to precision values.
        Default uses PRECISION settings based on key names.
    """
    if precision_map is None:
        precision_map = {}

    result = {}
    for key, value in d.items():
        # Determine precision based on key name
        if "mm" in key.lower():
            prec = PRECISION["mm"]
        elif "inch" in key.lower():
            prec = PRECISION["inch"]
        elif "position" in key.lower() or key in ("left", "bottom", "right", "top"):
            prec = PRECISION["position"]
        elif "lim" in key.lower():
            prec = PRECISION["lim"]
        elif "width" in key.lower() and "line" in key.lower():
            prec = PRECISION["linewidth"]
        else:
            prec = precision_map.get(key, 3)  # Default 3 decimals

        if isinstance(value, dict):
            result[key] = _round_dict(value, precision_map)
        elif isinstance(value, list):
            result[key] = _round_list(value, prec)
        elif isinstance(value, float):
            result[key] = _round_value(value, prec)
        else:
            result[key] = value

    return result


def _round_metadata(metadata: dict) -> dict:
    """
    Apply appropriate precision rounding to all numeric values in metadata.

    Precision rules:
    - mm values: 2 decimal places (0.01mm = 10 microns)
    - inch values: 3 decimal places
    - position values: 3 decimal places
    - axis limits: 2 decimal places
    - linewidth: 2 decimal places
    - px values: integers (no decimals)
    """
    result = {}

    for key, value in metadata.items():
        if key in ("scitex_schema", "scitex_schema_version", "figure_uuid"):
            # String fields - no rounding
            result[key] = value
        elif key == "runtime":
            # Runtime section - no numeric values to round
            result[key] = value
        elif key == "figure":
            result[key] = _round_figure_section(value)
        elif key == "axes":
            result[key] = _round_axes_section(value)
        elif key == "style":
            result[key] = _round_style_section(value)
        elif key == "plot":
            result[key] = _round_plot_section(value)
        elif key == "data":
            # Data section - no numeric values to round (hashes, paths, column names)
            result[key] = value
        elif key == "stats":
            # Stats section - preserve precision for statistical values
            result[key] = value
        else:
            result[key] = value

    return result
