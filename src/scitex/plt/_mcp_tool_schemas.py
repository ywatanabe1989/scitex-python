#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/plt/_mcp_tool_schemas.py
# ----------------------------------------

"""
MCP Tool schemas for SciTeX plt module.

Defines tools for publication-quality plotting:
- get_style: Get current style configuration
- set_style: Set global style overrides
- list_presets: List available style presets
- create_figure: Create figure with publication style
- crop_figure: Auto-crop figure whitespace
"""

from __future__ import annotations

import mcp.types as types


def get_tool_schemas() -> list[types.Tool]:
    """Return list of available MCP tools for plt operations."""
    return [
        # Get current style
        types.Tool(
            name="get_style",
            description="Get current SciTeX publication style configuration with all parameters (axes dimensions, fonts, margins, etc.)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # Set style
        types.Tool(
            name="set_style",
            description="Set global style overrides for publication figures. Parameters like axes_width_mm, margin_left_mm, font sizes, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "axes_width_mm": {
                        "type": "number",
                        "description": "Axes width in millimeters (default: 40)",
                    },
                    "axes_height_mm": {
                        "type": "number",
                        "description": "Axes height in millimeters (default: 28)",
                    },
                    "margin_left_mm": {
                        "type": "number",
                        "description": "Left margin in millimeters (default: 20)",
                    },
                    "margin_right_mm": {
                        "type": "number",
                        "description": "Right margin in millimeters (default: 20)",
                    },
                    "margin_top_mm": {
                        "type": "number",
                        "description": "Top margin in millimeters (default: 20)",
                    },
                    "margin_bottom_mm": {
                        "type": "number",
                        "description": "Bottom margin in millimeters (default: 20)",
                    },
                    "dpi": {
                        "type": "integer",
                        "description": "Output resolution in DPI (default: 300)",
                    },
                    "axis_font_size_pt": {
                        "type": "number",
                        "description": "Axis label font size in points (default: 7)",
                    },
                    "tick_font_size_pt": {
                        "type": "number",
                        "description": "Tick label font size in points (default: 7)",
                    },
                    "title_font_size_pt": {
                        "type": "number",
                        "description": "Title font size in points (default: 8)",
                    },
                    "legend_font_size_pt": {
                        "type": "number",
                        "description": "Legend font size in points (default: 6)",
                    },
                    "trace_thickness_mm": {
                        "type": "number",
                        "description": "Line trace thickness in mm (default: 0.2)",
                    },
                    "reset": {
                        "type": "boolean",
                        "description": "If true, reset style to defaults before applying",
                    },
                },
                "required": [],
            },
        ),
        # List presets
        types.Tool(
            name="list_presets",
            description="List available publication style presets (e.g., Nature, Science journal styles)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # Crop figure
        types.Tool(
            name="crop_figure",
            description="Auto-crop whitespace from a saved figure image",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the input figure image",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path (optional, adds '_cropped' suffix if not provided)",
                    },
                    "margin": {
                        "type": "integer",
                        "description": "Margin in pixels around content (default: 12, ~1mm at 300 DPI)",
                        "default": 12,
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite input file (default: false)",
                        "default": False,
                    },
                },
                "required": ["input_path"],
            },
        ),
        # Get DPI settings
        types.Tool(
            name="get_dpi_settings",
            description="Get DPI settings for different output contexts (save, display, preview)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # Get color palette
        types.Tool(
            name="get_color_palette",
            description="Get the SciTeX color palette for consistent figure colors",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Color format to return",
                        "enum": ["hex", "rgb", "rgba"],
                        "default": "hex",
                    },
                },
                "required": [],
            },
        ),
    ]


__all__ = ["get_tool_schemas"]

# EOF
