# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/tool_schemas.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-08
# # File: src/scitex/plt/_mcp.tool_schemas.py
# # ----------------------------------------
# 
# """
# MCP Tool schemas for SciTeX plt module.
# 
# Defines tools for publication-quality plotting:
# - get_style: Get current style configuration
# - set_style: Set global style overrides
# - list_presets: List available style presets
# - create_figure: Create figure with publication style
# - crop_figure: Auto-crop figure whitespace
# """
# 
# from __future__ import annotations
# 
# import mcp.types as types
# 
# 
# def get_tool_schemas() -> list[types.Tool]:
#     """Return list of available MCP tools for plt operations."""
#     return [
#         # Get current style
#         types.Tool(
#             name="get_style",
#             description="Get current SciTeX publication style configuration with all parameters (axes dimensions, fonts, margins, etc.)",
#             inputSchema={
#                 "type": "object",
#                 "properties": {},
#                 "required": [],
#             },
#         ),
#         # Set style
#         types.Tool(
#             name="set_style",
#             description="Set global style overrides for publication figures. Parameters like axes_width_mm, margin_left_mm, font sizes, etc.",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "axes_width_mm": {
#                         "type": "number",
#                         "description": "Axes width in millimeters (default: 40)",
#                     },
#                     "axes_height_mm": {
#                         "type": "number",
#                         "description": "Axes height in millimeters (default: 28)",
#                     },
#                     "margin_left_mm": {
#                         "type": "number",
#                         "description": "Left margin in millimeters (default: 20)",
#                     },
#                     "margin_right_mm": {
#                         "type": "number",
#                         "description": "Right margin in millimeters (default: 20)",
#                     },
#                     "margin_top_mm": {
#                         "type": "number",
#                         "description": "Top margin in millimeters (default: 20)",
#                     },
#                     "margin_bottom_mm": {
#                         "type": "number",
#                         "description": "Bottom margin in millimeters (default: 20)",
#                     },
#                     "dpi": {
#                         "type": "integer",
#                         "description": "Output resolution in DPI (default: 300)",
#                     },
#                     "axis_font_size_pt": {
#                         "type": "number",
#                         "description": "Axis label font size in points (default: 7)",
#                     },
#                     "tick_font_size_pt": {
#                         "type": "number",
#                         "description": "Tick label font size in points (default: 7)",
#                     },
#                     "title_font_size_pt": {
#                         "type": "number",
#                         "description": "Title font size in points (default: 8)",
#                     },
#                     "legend_font_size_pt": {
#                         "type": "number",
#                         "description": "Legend font size in points (default: 6)",
#                     },
#                     "trace_thickness_mm": {
#                         "type": "number",
#                         "description": "Line trace thickness in mm (default: 0.2)",
#                     },
#                     "reset": {
#                         "type": "boolean",
#                         "description": "If true, reset style to defaults before applying",
#                     },
#                 },
#                 "required": [],
#             },
#         ),
#         # List presets
#         types.Tool(
#             name="list_presets",
#             description="List available publication style presets (e.g., Nature, Science journal styles)",
#             inputSchema={
#                 "type": "object",
#                 "properties": {},
#                 "required": [],
#             },
#         ),
#         # Crop figure
#         types.Tool(
#             name="crop_figure",
#             description="Auto-crop whitespace from a saved figure image",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "input_path": {
#                         "type": "string",
#                         "description": "Path to the input figure image",
#                     },
#                     "output_path": {
#                         "type": "string",
#                         "description": "Output path (optional, adds '_cropped' suffix if not provided)",
#                     },
#                     "margin": {
#                         "type": "integer",
#                         "description": "Margin in pixels around content (default: 12, ~1mm at 300 DPI)",
#                         "default": 12,
#                     },
#                     "overwrite": {
#                         "type": "boolean",
#                         "description": "Overwrite input file (default: false)",
#                         "default": False,
#                     },
#                 },
#                 "required": ["input_path"],
#             },
#         ),
#         # Get DPI settings
#         types.Tool(
#             name="get_dpi_settings",
#             description="Get DPI settings for different output contexts (save, display, preview)",
#             inputSchema={
#                 "type": "object",
#                 "properties": {},
#                 "required": [],
#             },
#         ),
#         # Get color palette
#         types.Tool(
#             name="get_color_palette",
#             description="Get the SciTeX color palette for consistent figure colors",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "format": {
#                         "type": "string",
#                         "description": "Color format to return",
#                         "enum": ["hex", "rgb", "rgba"],
#                         "default": "hex",
#                     },
#                 },
#                 "required": [],
#             },
#         ),
#         # =================================================================
#         # PLOTTING TOOLS - Create publication figures via MCP
#         # =================================================================
#         # Create multi-panel figure
#         types.Tool(
#             name="create_figure",
#             description="Create a multi-panel figure canvas with SciTeX style. Returns figure_id for subsequent plotting.",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "nrows": {
#                         "type": "integer",
#                         "description": "Number of rows (default: 1)",
#                         "default": 1,
#                     },
#                     "ncols": {
#                         "type": "integer",
#                         "description": "Number of columns (default: 1)",
#                         "default": 1,
#                     },
#                     "axes_width_mm": {
#                         "type": "number",
#                         "description": "Width of each axes in mm (default: 40)",
#                         "default": 40,
#                     },
#                     "axes_height_mm": {
#                         "type": "number",
#                         "description": "Height of each axes in mm (default: 28)",
#                         "default": 28,
#                     },
#                     "space_w_mm": {
#                         "type": "number",
#                         "description": "Horizontal spacing between panels in mm (default: 8 from style)",
#                     },
#                     "space_h_mm": {
#                         "type": "number",
#                         "description": "Vertical spacing between panels in mm (default: 10 from style)",
#                     },
#                 },
#                 "required": [],
#             },
#         ),
#         # Bar plot
#         types.Tool(
#             name="plot_bar",
#             description="Create a bar plot on specified panel",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {
#                         "type": "string",
#                         "description": "Figure ID from create_figure (uses current if not specified)",
#                     },
#                     "panel": {
#                         "type": "string",
#                         "description": "Panel index as 'row,col' e.g. '0,0' or panel label e.g. 'A' (default: '0,0')",
#                         "default": "0,0",
#                     },
#                     "x": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "X-axis labels/categories",
#                     },
#                     "y": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "description": "Y-axis values (heights)",
#                     },
#                     "yerr": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "description": "Error bar values (optional)",
#                     },
#                     "colors": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "Bar colors (optional, uses palette)",
#                     },
#                     "xlabel": {"type": "string", "description": "X-axis label"},
#                     "ylabel": {"type": "string", "description": "Y-axis label"},
#                     "title": {"type": "string", "description": "Panel title"},
#                 },
#                 "required": ["x", "y"],
#             },
#         ),
#         # Scatter plot
#         types.Tool(
#             name="plot_scatter",
#             description="Create a scatter plot on specified panel",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {"type": "string", "description": "Figure ID"},
#                     "panel": {"type": "string", "default": "0,0"},
#                     "x": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "description": "X values",
#                     },
#                     "y": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "description": "Y values",
#                     },
#                     "color": {"type": "string", "description": "Point color"},
#                     "size": {"type": "number", "description": "Point size in mm"},
#                     "alpha": {"type": "number", "description": "Transparency (0-1)"},
#                     "add_regression": {
#                         "type": "boolean",
#                         "description": "Add regression line",
#                         "default": False,
#                     },
#                     "xlabel": {"type": "string"},
#                     "ylabel": {"type": "string"},
#                     "title": {"type": "string"},
#                 },
#                 "required": ["x", "y"],
#             },
#         ),
#         # Line plot
#         types.Tool(
#             name="plot_line",
#             description="Create a line plot on specified panel",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {"type": "string"},
#                     "panel": {"type": "string", "default": "0,0"},
#                     "x": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "description": "X values",
#                     },
#                     "y": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "description": "Y values",
#                     },
#                     "yerr": {
#                         "type": "array",
#                         "items": {"type": "number"},
#                         "description": "Y error values for shaded region",
#                     },
#                     "color": {"type": "string"},
#                     "label": {"type": "string", "description": "Legend label"},
#                     "linestyle": {
#                         "type": "string",
#                         "enum": ["-", "--", ":", "-."],
#                         "default": "-",
#                     },
#                     "xlabel": {"type": "string"},
#                     "ylabel": {"type": "string"},
#                     "title": {"type": "string"},
#                 },
#                 "required": ["x", "y"],
#             },
#         ),
#         # Box plot
#         types.Tool(
#             name="plot_box",
#             description="Create a box plot on specified panel",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {"type": "string"},
#                     "panel": {"type": "string", "default": "0,0"},
#                     "data": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "number"}},
#                         "description": "List of data arrays, one per group",
#                     },
#                     "labels": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "Group labels",
#                     },
#                     "colors": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "Box colors",
#                     },
#                     "xlabel": {"type": "string"},
#                     "ylabel": {"type": "string"},
#                     "title": {"type": "string"},
#                 },
#                 "required": ["data"],
#             },
#         ),
#         # Violin plot
#         types.Tool(
#             name="plot_violin",
#             description="Create a violin plot on specified panel",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {"type": "string"},
#                     "panel": {"type": "string", "default": "0,0"},
#                     "data": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "number"}},
#                         "description": "List of data arrays, one per group",
#                     },
#                     "labels": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                     },
#                     "colors": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                     },
#                     "xlabel": {"type": "string"},
#                     "ylabel": {"type": "string"},
#                     "title": {"type": "string"},
#                 },
#                 "required": ["data"],
#             },
#         ),
#         # Add significance bracket
#         types.Tool(
#             name="add_significance",
#             description="Add significance bracket between two groups",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {"type": "string"},
#                     "panel": {"type": "string", "default": "0,0"},
#                     "x1": {
#                         "type": "number",
#                         "description": "X position of first group (0-indexed)",
#                     },
#                     "x2": {
#                         "type": "number",
#                         "description": "X position of second group",
#                     },
#                     "y": {
#                         "type": "number",
#                         "description": "Y position of bracket base",
#                     },
#                     "text": {
#                         "type": "string",
#                         "description": "Significance text: '*', '**', '***', 'n.s.', or p-value",
#                     },
#                     "height": {
#                         "type": "number",
#                         "description": "Bracket height (auto if not specified)",
#                     },
#                 },
#                 "required": ["x1", "x2", "y", "text"],
#             },
#         ),
#         # Add panel label
#         types.Tool(
#             name="add_panel_label",
#             description="Add panel label (A, B, C, etc.) to a panel",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {"type": "string"},
#                     "panel": {"type": "string", "default": "0,0"},
#                     "label": {
#                         "type": "string",
#                         "description": "Label text (e.g., 'A', 'B')",
#                     },
#                     "x": {
#                         "type": "number",
#                         "description": "X position in axes coordinates (default: -0.15)",
#                         "default": -0.15,
#                     },
#                     "y": {
#                         "type": "number",
#                         "description": "Y position in axes coordinates (default: 1.1)",
#                         "default": 1.1,
#                     },
#                     "fontsize": {
#                         "type": "number",
#                         "description": "Font size in points (default: 10)",
#                         "default": 10,
#                     },
#                     "fontweight": {
#                         "type": "string",
#                         "enum": ["normal", "bold"],
#                         "default": "bold",
#                     },
#                 },
#                 "required": ["label"],
#             },
#         ),
#         # Save figure
#         types.Tool(
#             name="save_figure",
#             description="Save the current figure to file",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {"type": "string"},
#                     "output_path": {
#                         "type": "string",
#                         "description": "Output file path (supports .png, .pdf, .svg)",
#                     },
#                     "dpi": {
#                         "type": "integer",
#                         "description": "Resolution (default: 300)",
#                         "default": 300,
#                     },
#                     "crop": {
#                         "type": "boolean",
#                         "description": "Auto-crop whitespace (default: true)",
#                         "default": True,
#                     },
#                 },
#                 "required": ["output_path"],
#             },
#         ),
#         # Close figure
#         types.Tool(
#             name="close_figure",
#             description="Close a figure and free memory",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "figure_id": {
#                         "type": "string",
#                         "description": "Figure ID to close (closes current if not specified)",
#                     },
#                 },
#                 "required": [],
#             },
#         ),
#     ]
# 
# 
# __all__ = ["get_tool_schemas"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/tool_schemas.py
# --------------------------------------------------------------------------------
