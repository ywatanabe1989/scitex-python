#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/canvas/_mcp_tool_schemas.py
# ----------------------------------------

"""
MCP Tool schemas for SciTeX canvas module.

Defines tools for multi-panel figure composition:
- create_canvas: Create paper figure workspace
- add_panel: Add subplot to canvas
- list_panels: List panels in canvas
- export_canvas: Render final figure
- list_canvases: List available canvases
"""

from __future__ import annotations

import mcp.types as types


def get_tool_schemas() -> list[types.Tool]:
    """Return list of available MCP tools for canvas operations."""
    return [
        # Create canvas
        types.Tool(
            name="create_canvas",
            description="Create a new paper figure canvas workspace for multi-panel composition",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_dir": {
                        "type": "string",
                        "description": "Parent directory for the canvas",
                    },
                    "canvas_name": {
                        "type": "string",
                        "description": "Name for the canvas (e.g., 'fig1', 'Figure_2')",
                    },
                    "width_mm": {
                        "type": "number",
                        "description": "Canvas width in millimeters (default: 180 for double column)",
                        "default": 180,
                    },
                    "height_mm": {
                        "type": "number",
                        "description": "Canvas height in millimeters (default: 120)",
                        "default": 120,
                    },
                },
                "required": ["parent_dir", "canvas_name"],
            },
        ),
        # Add panel
        types.Tool(
            name="add_panel",
            description="Add a panel (subplot) to an existing canvas from an image or plot file",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_dir": {
                        "type": "string",
                        "description": "Parent directory containing the canvas",
                    },
                    "canvas_name": {
                        "type": "string",
                        "description": "Canvas name",
                    },
                    "panel_name": {
                        "type": "string",
                        "description": "Name for this panel (e.g., 'panel_a', 'timecourse')",
                    },
                    "source": {
                        "type": "string",
                        "description": "Path to source file (PNG, JPG, SVG, or SciTeX plot)",
                    },
                    "x_mm": {
                        "type": "number",
                        "description": "X position in millimeters from left edge",
                        "default": 0,
                    },
                    "y_mm": {
                        "type": "number",
                        "description": "Y position in millimeters from top edge",
                        "default": 0,
                    },
                    "width_mm": {
                        "type": "number",
                        "description": "Panel width in millimeters",
                        "default": 50,
                    },
                    "height_mm": {
                        "type": "number",
                        "description": "Panel height in millimeters",
                        "default": 50,
                    },
                    "label": {
                        "type": "string",
                        "description": "Panel label (e.g., 'A', 'B', 'C')",
                    },
                },
                "required": ["parent_dir", "canvas_name", "panel_name", "source"],
            },
        ),
        # List panels
        types.Tool(
            name="list_panels",
            description="List all panels in a canvas with their properties",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_dir": {
                        "type": "string",
                        "description": "Parent directory containing the canvas",
                    },
                    "canvas_name": {
                        "type": "string",
                        "description": "Canvas name",
                    },
                },
                "required": ["parent_dir", "canvas_name"],
            },
        ),
        # Remove panel
        types.Tool(
            name="remove_panel",
            description="Remove a panel from a canvas",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_dir": {
                        "type": "string",
                        "description": "Parent directory containing the canvas",
                    },
                    "canvas_name": {
                        "type": "string",
                        "description": "Canvas name",
                    },
                    "panel_name": {
                        "type": "string",
                        "description": "Name of the panel to remove",
                    },
                },
                "required": ["parent_dir", "canvas_name", "panel_name"],
            },
        ),
        # Export canvas
        types.Tool(
            name="export_canvas",
            description="Export/render canvas to PNG, PDF, or SVG format",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_dir": {
                        "type": "string",
                        "description": "Parent directory containing the canvas",
                    },
                    "canvas_name": {
                        "type": "string",
                        "description": "Canvas name",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (format determined by extension)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format (auto-detected from path if not specified)",
                        "enum": ["png", "pdf", "svg"],
                    },
                    "dpi": {
                        "type": "integer",
                        "description": "Output DPI for raster formats (default: 300)",
                        "default": 300,
                    },
                },
                "required": ["parent_dir", "canvas_name"],
            },
        ),
        # List canvases
        types.Tool(
            name="list_canvases",
            description="List all canvases in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_dir": {
                        "type": "string",
                        "description": "Directory to search for canvases",
                    },
                },
                "required": ["parent_dir"],
            },
        ),
        # Canvas exists
        types.Tool(
            name="canvas_exists",
            description="Check if a canvas exists",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_dir": {
                        "type": "string",
                        "description": "Parent directory",
                    },
                    "canvas_name": {
                        "type": "string",
                        "description": "Canvas name to check",
                    },
                },
                "required": ["parent_dir", "canvas_name"],
            },
        ),
    ]


__all__ = ["get_tool_schemas"]

# EOF
