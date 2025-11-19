#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-19 13:45:00 (ywatanabe)"
# File: ./src/scitex/plt/presets.py

"""
Standard style presets for scitex.plt.subplots().

Usage:
    from scitex.plt.presets import NATURE_STYLE, SCIENCE_STYLE

    fig, ax = stx.plt.subplots(**NATURE_STYLE)

    # Override specific parameters
    style = NATURE_STYLE.copy()
    style['ax_width_mm'] = 40
    fig, ax = stx.plt.subplots(**style)
"""

__all__ = [
    "NATURE_STYLE",
    "SCIENCE_STYLE",
    "CELL_STYLE",
    "PNAS_STYLE",
]

# Nature journal style
# Based on Nature's figure requirements:
# - Single column: 89 mm (3.5 inches)
# - Double column: 183 mm (7.2 inches)
# - Text: 8pt for axis labels, 7pt for tick labels
NATURE_STYLE = {
    "ax_width_mm": 30,
    "ax_height_mm": 21,
    "ax_thickness_mm": 0.2,
    "tick_length_mm": 0.8,
    "tick_thickness_mm": 0.2,
    "trace_thickness_mm": 0.12,
    "axis_font_size_pt": 8,
    "tick_font_size_pt": 7,
    "margin_left_mm": 5,
    "margin_right_mm": 2,
    "margin_bottom_mm": 5,
    "margin_top_mm": 2,
    "space_w_mm": 3,
    "space_h_mm": 3,
    "mode": "publication",
    "dpi": 300,
}

# Science journal style
# Based on Science's figure requirements:
# - Single column: 90 mm (3.54 inches)
# - Double column: 183 mm (7.2 inches)
SCIENCE_STYLE = {
    "ax_width_mm": 35,
    "ax_height_mm": 24.5,
    "ax_thickness_mm": 0.25,
    "tick_length_mm": 1.0,
    "tick_thickness_mm": 0.25,
    "trace_thickness_mm": 0.15,
    "axis_font_size_pt": 8,
    "tick_font_size_pt": 7,
    "margin_left_mm": 5,
    "margin_right_mm": 2,
    "margin_bottom_mm": 5,
    "margin_top_mm": 2,
    "space_w_mm": 3,
    "space_h_mm": 3,
    "mode": "publication",
    "dpi": 300,
}

# Cell journal style
# Based on Cell Press requirements:
# - Single column: 85 mm (3.35 inches)
# - Double column: 174 mm (6.85 inches)
CELL_STYLE = {
    "ax_width_mm": 32,
    "ax_height_mm": 22,
    "ax_thickness_mm": 0.25,
    "tick_length_mm": 1.0,
    "tick_thickness_mm": 0.25,
    "trace_thickness_mm": 0.15,
    "axis_font_size_pt": 8,
    "tick_font_size_pt": 7,
    "margin_left_mm": 5,
    "margin_right_mm": 2,
    "margin_bottom_mm": 5,
    "margin_top_mm": 2,
    "space_w_mm": 3,
    "space_h_mm": 3,
    "mode": "publication",
    "dpi": 300,
}

# PNAS journal style
# Based on PNAS figure requirements:
# - Single column: 87 mm (3.42 inches)
# - Double column: 178 mm (7.0 inches)
PNAS_STYLE = {
    "ax_width_mm": 33,
    "ax_height_mm": 23,
    "ax_thickness_mm": 0.25,
    "tick_length_mm": 1.0,
    "tick_thickness_mm": 0.25,
    "trace_thickness_mm": 0.15,
    "axis_font_size_pt": 8,
    "tick_font_size_pt": 7,
    "margin_left_mm": 5,
    "margin_right_mm": 2,
    "margin_bottom_mm": 5,
    "margin_top_mm": 2,
    "space_w_mm": 3,
    "space_h_mm": 3,
    "mode": "publication",
    "dpi": 300,
}

# EOF
