#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-19 13:45:00 (ywatanabe)"
# File: ./src/scitex/plt/presets.py

"""
Standard style presets for scitex.plt.subplots().

Usage:
    from scitex.plt.presets import SCITEX_STYLE

    # Use the universal SciTeX preset (recommended)
    fig, ax = stx.plt.subplots(**SCITEX_STYLE)

    # Or use journal-specific presets
    from scitex.plt.presets import NATURE_STYLE, SCIENCE_STYLE
    fig, ax = stx.plt.subplots(**NATURE_STYLE)

    # Override specific parameters
    style = SCITEX_STYLE.copy()
    style['ax_width_mm'] = 50
    fig, ax = stx.plt.subplots(**style)
"""

__all__ = [
    "SCITEX_STYLE",
    "NATURE_STYLE",
    "SCIENCE_STYLE",
    "CELL_STYLE",
    "PNAS_STYLE",
]

# SciTeX default style
# Universal publication-ready preset suitable for most journals
# Based on common requirements across Nature, Science, Cell, PNAS
SCITEX_STYLE = {
    "ax_width_mm": 40,
    "ax_height_mm": 28,  # 1:0.7 ratio (width:height) for most plots
    "ax_thickness_mm": 0.2,
    "tick_length_mm": 0.8,
    "tick_thickness_mm": 0.2,
    "trace_thickness_mm": 0.2,  # Regular line plots (use 0.12mm for signals like EEG)
    "errorbar_thickness_mm": 0.2,  # Error bar line thickness
    "errorbar_cap_width_mm": 0.8,  # Error bar cap width (0.8mm)
    "bar_edge_thickness_mm": 0.2,  # Bar plot edge thickness
    "kde_line_thickness_mm": 0.2,  # KDE line thickness
    "scatter_size_mm": 0.8,  # Scatter marker size
    "axis_font_size_pt": 7,
    "tick_font_size_pt": 7,
    "title_font_size_pt": 8,  # Slightly larger than axis labels (Nature-style)
    "suptitle_font_size_pt": 8,
    "legend_font_size_pt": 6,
    "label_pad_pt": 0.5,      # Axis label to axis distance (Nature-style: extremely tight, default is 4pt)
    "tick_pad_pt": 2.0,       # Tick label to tick distance (readable spacing, default is 3pt)
    "title_pad_pt": 1.0,      # Title to axis top distance (Nature-style: extremely tight, default is ~6pt)
    "font_family": "Arial",   # Arial font for publications
    "margin_left_mm": 20,    # Large margins for labels, crop afterwards
    "margin_right_mm": 20,   # Large margins for colorbars, crop afterwards
    "margin_bottom_mm": 20,  # Large margins for x-axis labels, crop afterwards
    "margin_top_mm": 20,     # Large margins for titles, crop afterwards
    "space_w_mm": 8,         # Large spacing for multi-panel, crop afterwards
    "space_h_mm": 10,        # Large spacing for multi-panel, crop afterwards
    "n_ticks": 4,            # Target 3-4 ticks on each axis
    "transparent": True,     # Transparent background for professional crop workflow
    "mode": "publication",
    "dpi": 300,
}

# Nature journal style
# Based on Nature's figure requirements:
# - Single column: 89 mm (3.5 inches)
# - Double column: 183 mm (7.2 inches)
# - Text: 8pt for axis labels, 7pt for tick labels
NATURE_STYLE = {
    "ax_width_mm": 40,
    "ax_height_mm": 28,  # 1:0.7 ratio (width:height) for most plots
    "ax_thickness_mm": 0.2,
    "tick_length_mm": 0.8,
    "tick_thickness_mm": 0.2,
    "trace_thickness_mm": 0.2,  # Regular line plots (use 0.12mm for signals like EEG)
    "errorbar_thickness_mm": 0.2,  # Error bar line thickness
    "errorbar_cap_width_mm": 0.8,  # Error bar cap width (0.8mm)
    "bar_edge_thickness_mm": 0.2,  # Bar plot edge thickness
    "kde_line_thickness_mm": 0.2,  # KDE line thickness
    "scatter_size_mm": 0.8,  # Scatter marker size
    "errorbar_thickness_mm": 0.2,  # Error bar line thickness
    "errorbar_cap_width_mm": 0.8,  # Error bar cap width (0.8mm)
    "bar_edge_thickness_mm": 0.2,  # Bar plot edge thickness
    "kde_line_thickness_mm": 0.2,  # KDE line thickness
    "scatter_size_mm": 0.8,  # Scatter marker size
    "axis_font_size_pt": 7,
    "tick_font_size_pt": 7,
    "title_font_size_pt": 8,  # Slightly larger than axis labels (Nature-style)
    "suptitle_font_size_pt": 8,
    "legend_font_size_pt": 6,
    "label_pad_pt": 0.5,      # Axis label to axis distance (Nature-style: extremely tight, default is 4pt)
    "tick_pad_pt": 2.0,       # Tick label to tick distance (readable spacing, default is 3pt)
    "title_pad_pt": 1.0,      # Title to axis top distance (Nature-style: extremely tight, default is ~6pt)
    "margin_left_mm": 20,    # Large margins for labels, crop afterwards
    "margin_right_mm": 20,   # Large margins for colorbars, crop afterwards
    "margin_bottom_mm": 20,  # Large margins for x-axis labels, crop afterwards
    "margin_top_mm": 20,     # Large margins for titles, crop afterwards
    "space_w_mm": 8,         # Large spacing for multi-panel, crop afterwards
    "space_h_mm": 10,        # Large spacing for multi-panel, crop afterwards
    "n_ticks": 4,            # Target 3-4 ticks on each axis
    "transparent": True,     # Transparent background for professional crop workflow
    "mode": "publication",
    "dpi": 300,
}

# Science journal style
# Based on Science's figure requirements:
# - Single column: 90 mm (3.54 inches)
# - Double column: 183 mm (7.2 inches)
SCIENCE_STYLE = {
    "ax_width_mm": 45,
    "ax_height_mm": 31.5,  # 1:0.7 ratio (width:height) for most plots
    "ax_thickness_mm": 0.25,
    "tick_length_mm": 1.0,
    "tick_thickness_mm": 0.25,
    "trace_thickness_mm": 0.15,
    "errorbar_thickness_mm": 0.2,  # Error bar line thickness
    "errorbar_cap_width_mm": 0.8,  # Error bar cap width (0.8mm)
    "bar_edge_thickness_mm": 0.2,  # Bar plot edge thickness
    "kde_line_thickness_mm": 0.2,  # KDE line thickness
    "scatter_size_mm": 0.8,  # Scatter marker size
    "axis_font_size_pt": 7,
    "tick_font_size_pt": 7,
    "title_font_size_pt": 8,  # Slightly larger than axis labels (Nature-style)
    "suptitle_font_size_pt": 8,
    "legend_font_size_pt": 6,
    "label_pad_pt": 0.5,      # Axis label to axis distance (Nature-style: extremely tight, default is 4pt)
    "tick_pad_pt": 2.0,       # Tick label to tick distance (readable spacing, default is 3pt)
    "title_pad_pt": 1.0,      # Title to axis top distance (Nature-style: extremely tight, default is ~6pt)
    "margin_left_mm": 20,    # Large margins for labels, crop afterwards
    "margin_right_mm": 20,   # Large margins for colorbars, crop afterwards
    "margin_bottom_mm": 20,  # Large margins for x-axis labels, crop afterwards
    "margin_top_mm": 20,     # Large margins for titles, crop afterwards
    "space_w_mm": 8,         # Large spacing for multi-panel, crop afterwards
    "space_h_mm": 10,        # Large spacing for multi-panel, crop afterwards
    "n_ticks": 4,            # Target 3-4 ticks on each axis
    "transparent": True,     # Transparent background for professional crop workflow
    "mode": "publication",
    "dpi": 300,
}

# Cell journal style
# Based on Cell Press requirements:
# - Single column: 85 mm (3.35 inches)
# - Double column: 174 mm (6.85 inches)
CELL_STYLE = {
    "ax_width_mm": 42,
    "ax_height_mm": 29.4,  # 1:0.7 ratio (width:height) for most plots
    "ax_thickness_mm": 0.25,
    "tick_length_mm": 1.0,
    "tick_thickness_mm": 0.25,
    "trace_thickness_mm": 0.15,
    "errorbar_thickness_mm": 0.2,  # Error bar line thickness
    "errorbar_cap_width_mm": 0.8,  # Error bar cap width (0.8mm)
    "bar_edge_thickness_mm": 0.2,  # Bar plot edge thickness
    "kde_line_thickness_mm": 0.2,  # KDE line thickness
    "scatter_size_mm": 0.8,  # Scatter marker size
    "axis_font_size_pt": 7,
    "tick_font_size_pt": 7,
    "title_font_size_pt": 8,  # Slightly larger than axis labels (Nature-style)
    "suptitle_font_size_pt": 8,
    "legend_font_size_pt": 6,
    "label_pad_pt": 0.5,      # Axis label to axis distance (Nature-style: extremely tight, default is 4pt)
    "tick_pad_pt": 2.0,       # Tick label to tick distance (readable spacing, default is 3pt)
    "title_pad_pt": 1.0,      # Title to axis top distance (Nature-style: extremely tight, default is ~6pt)
    "margin_left_mm": 20,    # Large margins for labels, crop afterwards
    "margin_right_mm": 20,   # Large margins for colorbars, crop afterwards
    "margin_bottom_mm": 20,  # Large margins for x-axis labels, crop afterwards
    "margin_top_mm": 20,     # Large margins for titles, crop afterwards
    "space_w_mm": 8,         # Large spacing for multi-panel, crop afterwards
    "space_h_mm": 10,        # Large spacing for multi-panel, crop afterwards
    "n_ticks": 4,            # Target 3-4 ticks on each axis
    "transparent": True,     # Transparent background for professional crop workflow
    "mode": "publication",
    "dpi": 300,
}

# PNAS journal style
# Based on PNAS figure requirements:
# - Single column: 87 mm (3.42 inches)
# - Double column: 178 mm (7.0 inches)
PNAS_STYLE = {
    "ax_width_mm": 43,
    "ax_height_mm": 30.1,  # 1:0.7 ratio (width:height) for most plots
    "ax_thickness_mm": 0.25,
    "tick_length_mm": 1.0,
    "tick_thickness_mm": 0.25,
    "trace_thickness_mm": 0.15,
    "errorbar_thickness_mm": 0.2,  # Error bar line thickness
    "errorbar_cap_width_mm": 0.8,  # Error bar cap width (0.8mm)
    "bar_edge_thickness_mm": 0.2,  # Bar plot edge thickness
    "kde_line_thickness_mm": 0.2,  # KDE line thickness
    "scatter_size_mm": 0.8,  # Scatter marker size
    "axis_font_size_pt": 7,
    "tick_font_size_pt": 7,
    "title_font_size_pt": 8,  # Slightly larger than axis labels (Nature-style)
    "suptitle_font_size_pt": 8,
    "legend_font_size_pt": 6,
    "label_pad_pt": 0.5,      # Axis label to axis distance (Nature-style: extremely tight, default is 4pt)
    "tick_pad_pt": 2.0,       # Tick label to tick distance (readable spacing, default is 3pt)
    "title_pad_pt": 1.0,      # Title to axis top distance (Nature-style: extremely tight, default is ~6pt)
    "margin_left_mm": 20,    # Large margins for labels, crop afterwards
    "margin_right_mm": 20,   # Large margins for colorbars, crop afterwards
    "margin_bottom_mm": 20,  # Large margins for x-axis labels, crop afterwards
    "margin_top_mm": 20,     # Large margins for titles, crop afterwards
    "space_w_mm": 8,         # Large spacing for multi-panel, crop afterwards
    "space_h_mm": 10,        # Large spacing for multi-panel, crop afterwards
    "n_ticks": 4,            # Target 3-4 ticks on each axis
    "transparent": True,     # Transparent background for professional crop workflow
    "mode": "publication",
    "dpi": 300,
}

# EOF
