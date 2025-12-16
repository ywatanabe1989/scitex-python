#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 21:00:00 (ywatanabe)"
# File: ./src/scitex/plt/styles/__init__.py

"""SciTeX plot styling module.

This module centralizes all plot-specific default styling, including:
- Pre-processing: Default kwargs applied before matplotlib method calls
- Post-processing: Styling applied after matplotlib method calls
- Style configuration with priority resolution: direct → yaml → env → default

Usage:
    from scitex.plt.styles import apply_plot_defaults, apply_plot_postprocess

    # In AxisWrapper.__getattr__ wrapper:
    apply_plot_defaults(method_name, kwargs, id_value, ax)
    result = orig_method(*args, **kwargs)
    apply_plot_postprocess(method_name, result, ax, kwargs)

    # Style configuration
    from scitex.plt.styles import SCITEX_STYLE, load_style
    fig, ax = stx.plt.subplots(**SCITEX_STYLE)

    # Custom YAML
    style = load_style("path/to/my_style.yaml")
    fig, ax = stx.plt.subplots(**style)
"""

from ._plot_defaults import apply_plot_defaults
from ._plot_postprocess import apply_plot_postprocess
from .presets import (
    SCITEX_STYLE,
    STYLE,
    load_style,
    save_style,
    set_style,
    get_style,
    resolve_style_value,
    # DPI utilities
    get_default_dpi,
    get_display_dpi,
    get_preview_dpi,
    DPI_SAVE,
    DPI_DISPLAY,
    DPI_PREVIEW,
)

__all__ = [
    # Styling functions
    "apply_plot_defaults",
    "apply_plot_postprocess",
    # Style configuration
    "SCITEX_STYLE",
    "STYLE",
    "load_style",
    "save_style",
    "set_style",
    "get_style",
    "resolve_style_value",
    # DPI utilities
    "get_default_dpi",
    "get_display_dpi",
    "get_preview_dpi",
    "DPI_SAVE",
    "DPI_DISPLAY",
    "DPI_PREVIEW",
]


# EOF
