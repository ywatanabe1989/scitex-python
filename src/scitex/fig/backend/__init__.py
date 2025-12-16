#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/backend/__init__.py
"""
Backend for rendering figure JSON to matplotlib figures.

This module bridges the gap between JSON specifications and
actual matplotlib figures using scitex.plt.
"""

from ._parser import (
    parse_figure_json,
    parse_axes_json,
    parse_plot_json,
    parse_guide_json,
    parse_annotation_json,
    validate_figure_json,
)

from ._render import (
    render_figure,
    render_axes,
    render_plot,
    render_guide,
    render_annotation,
    build_figure_from_json,
)

from ._export import (
    export_figure,
    export_figure_from_file,
    export_multiple_formats,
)

__all__ = [
    # Parser
    "parse_figure_json",
    "parse_axes_json",
    "parse_plot_json",
    "parse_guide_json",
    "parse_annotation_json",
    "validate_figure_json",
    # Renderer
    "render_figure",
    "render_axes",
    "render_plot",
    "render_guide",
    "render_annotation",
    "build_figure_from_json",
    # Exporter
    "export_figure",
    "export_figure_from_file",
    "export_multiple_formats",
]

# EOF
