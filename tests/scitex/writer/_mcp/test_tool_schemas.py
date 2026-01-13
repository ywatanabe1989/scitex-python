# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_mcp/tool_schemas.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-09
# # File: src/scitex/writer/_mcp.tool_schemas.py
# # ----------------------------------------
# 
# """
# MCP Tool schemas for SciTeX Writer module.
# 
# Defines available tools for LaTeX manuscript compilation:
# - clone_project: Create new writer project from template
# - compile_manuscript: Compile manuscript to PDF
# - compile_supplementary: Compile supplementary materials to PDF
# - compile_revision: Compile revision document with optional change tracking
# - get_project_info: Get project structure and status
# - get_pdf: Get path to compiled PDF
# """
# 
# from __future__ import annotations
# 
# import mcp.types as types
# 
# 
# def get_tool_schemas() -> list[types.Tool]:
#     """Return list of available MCP tools for writer operations."""
#     return [
#         # Clone writer project
#         types.Tool(
#             name="clone_project",
#             description="Create a new LaTeX manuscript project from template with proper directory structure",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "project_dir": {
#                         "type": "string",
#                         "description": "Path to create project directory",
#                     },
#                     "git_strategy": {
#                         "type": "string",
#                         "description": "Git initialization strategy",
#                         "enum": ["child", "parent", "origin", "none"],
#                         "default": "child",
#                     },
#                     "branch": {
#                         "type": "string",
#                         "description": "Specific branch of template to clone (optional)",
#                     },
#                     "tag": {
#                         "type": "string",
#                         "description": "Specific tag/release of template to clone (optional)",
#                     },
#                 },
#                 "required": ["project_dir"],
#             },
#         ),
#         # Compile manuscript
#         types.Tool(
#             name="compile_manuscript",
#             description="Compile manuscript LaTeX document to PDF",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "project_dir": {
#                         "type": "string",
#                         "description": "Path to writer project directory",
#                     },
#                     "timeout": {
#                         "type": "integer",
#                         "description": "Maximum compilation time in seconds (default: 300)",
#                         "default": 300,
#                     },
#                     "no_figs": {
#                         "type": "boolean",
#                         "description": "Exclude figures for quick compilation (default: false)",
#                         "default": False,
#                     },
#                     "ppt2tif": {
#                         "type": "boolean",
#                         "description": "Convert PowerPoint files to TIF format (WSL only, default: false)",
#                         "default": False,
#                     },
#                     "crop_tif": {
#                         "type": "boolean",
#                         "description": "Crop TIF images to remove excess whitespace (default: false)",
#                         "default": False,
#                     },
#                     "quiet": {
#                         "type": "boolean",
#                         "description": "Suppress detailed LaTeX compilation logs (default: false)",
#                         "default": False,
#                     },
#                     "verbose": {
#                         "type": "boolean",
#                         "description": "Show verbose LaTeX compilation output (default: false)",
#                         "default": False,
#                     },
#                     "force": {
#                         "type": "boolean",
#                         "description": "Force recompilation, ignore cache (default: false)",
#                         "default": False,
#                     },
#                 },
#                 "required": ["project_dir"],
#             },
#         ),
#         # Compile supplementary
#         types.Tool(
#             name="compile_supplementary",
#             description="Compile supplementary materials LaTeX document to PDF",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "project_dir": {
#                         "type": "string",
#                         "description": "Path to writer project directory",
#                     },
#                     "timeout": {
#                         "type": "integer",
#                         "description": "Maximum compilation time in seconds (default: 300)",
#                         "default": 300,
#                     },
#                     "no_figs": {
#                         "type": "boolean",
#                         "description": "Exclude figures for quick compilation (default: false)",
#                         "default": False,
#                     },
#                     "ppt2tif": {
#                         "type": "boolean",
#                         "description": "Convert PowerPoint files to TIF format (WSL only, default: false)",
#                         "default": False,
#                     },
#                     "crop_tif": {
#                         "type": "boolean",
#                         "description": "Crop TIF images to remove excess whitespace (default: false)",
#                         "default": False,
#                     },
#                     "quiet": {
#                         "type": "boolean",
#                         "description": "Suppress detailed LaTeX compilation logs (default: false)",
#                         "default": False,
#                     },
#                 },
#                 "required": ["project_dir"],
#             },
#         ),
#         # Compile revision
#         types.Tool(
#             name="compile_revision",
#             description="Compile revision document to PDF with optional change tracking",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "project_dir": {
#                         "type": "string",
#                         "description": "Path to writer project directory",
#                     },
#                     "track_changes": {
#                         "type": "boolean",
#                         "description": "Enable change tracking in output PDF (default: false)",
#                         "default": False,
#                     },
#                     "timeout": {
#                         "type": "integer",
#                         "description": "Maximum compilation time in seconds (default: 300)",
#                         "default": 300,
#                     },
#                 },
#                 "required": ["project_dir"],
#             },
#         ),
#         # Get project info
#         types.Tool(
#             name="get_project_info",
#             description="Get writer project structure and status information",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "project_dir": {
#                         "type": "string",
#                         "description": "Path to writer project directory",
#                     },
#                 },
#                 "required": ["project_dir"],
#             },
#         ),
#         # Get PDF path
#         types.Tool(
#             name="get_pdf",
#             description="Get path to compiled PDF for a document type",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "project_dir": {
#                         "type": "string",
#                         "description": "Path to writer project directory",
#                     },
#                     "doc_type": {
#                         "type": "string",
#                         "description": "Document type",
#                         "enum": ["manuscript", "supplementary", "revision"],
#                         "default": "manuscript",
#                     },
#                 },
#                 "required": ["project_dir"],
#             },
#         ),
#         # List document types
#         types.Tool(
#             name="list_document_types",
#             description="List available document types in a writer project",
#             inputSchema={
#                 "type": "object",
#                 "properties": {},
#                 "required": [],
#             },
#         ),
#         # CSV to LaTeX conversion
#         types.Tool(
#             name="csv_to_latex",
#             description="Convert CSV file to LaTeX table format",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "csv_path": {
#                         "type": "string",
#                         "description": "Path to CSV file",
#                     },
#                     "output_path": {
#                         "type": "string",
#                         "description": "Output path for LaTeX file (optional)",
#                     },
#                     "caption": {
#                         "type": "string",
#                         "description": "Table caption (optional)",
#                     },
#                     "label": {
#                         "type": "string",
#                         "description": "Table label for referencing (optional)",
#                     },
#                     "longtable": {
#                         "type": "boolean",
#                         "description": "Use longtable for multi-page tables (default: false)",
#                         "default": False,
#                     },
#                 },
#                 "required": ["csv_path"],
#             },
#         ),
#         # LaTeX to CSV conversion
#         types.Tool(
#             name="latex_to_csv",
#             description="Convert LaTeX table to CSV format",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "latex_path": {
#                         "type": "string",
#                         "description": "Path to LaTeX file containing table",
#                     },
#                     "output_path": {
#                         "type": "string",
#                         "description": "Output path for CSV file (optional)",
#                     },
#                     "table_index": {
#                         "type": "integer",
#                         "description": "Index of table to extract if multiple exist (default: 0)",
#                         "default": 0,
#                     },
#                 },
#                 "required": ["latex_path"],
#             },
#         ),
#         # PDF to image conversion
#         types.Tool(
#             name="pdf_to_images",
#             description="Render PDF pages as images. Useful for creating figures from PDF or extracting specific pages.",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "pdf_path": {
#                         "type": "string",
#                         "description": "Path to PDF file",
#                     },
#                     "output_dir": {
#                         "type": "string",
#                         "description": "Output directory for images (uses temp if not specified)",
#                     },
#                     "pages": {
#                         "oneOf": [
#                             {"type": "integer"},
#                             {"type": "array", "items": {"type": "integer"}},
#                         ],
#                         "description": "Page(s) to render (0-indexed). If not specified, renders all pages.",
#                     },
#                     "dpi": {
#                         "type": "integer",
#                         "description": "Resolution in DPI (default: 150)",
#                         "default": 150,
#                     },
#                     "format": {
#                         "type": "string",
#                         "description": "Output format",
#                         "enum": ["png", "jpg"],
#                         "default": "png",
#                     },
#                 },
#                 "required": ["pdf_path"],
#             },
#         ),
#         # List figures
#         types.Tool(
#             name="list_figures",
#             description="List all figures in a writer project directory",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "project_dir": {
#                         "type": "string",
#                         "description": "Path to writer project directory",
#                     },
#                     "extensions": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "File extensions to include (e.g., ['.png', '.pdf']). Uses common formats if not specified.",
#                     },
#                 },
#                 "required": ["project_dir"],
#             },
#         ),
#         # Convert figure format
#         types.Tool(
#             name="convert_figure",
#             description="Convert figure between formats (e.g., PDF to PNG, PNG to JPG)",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "input_path": {
#                         "type": "string",
#                         "description": "Input figure path",
#                     },
#                     "output_path": {
#                         "type": "string",
#                         "description": "Output figure path (format determined by extension)",
#                     },
#                     "dpi": {
#                         "type": "integer",
#                         "description": "Resolution for PDF rasterization (default: 300)",
#                         "default": 300,
#                     },
#                     "quality": {
#                         "type": "integer",
#                         "description": "JPEG quality 1-100 (default: 95)",
#                         "default": 95,
#                     },
#                 },
#                 "required": ["input_path", "output_path"],
#             },
#         ),
#     ]
# 
# 
# __all__ = ["get_tool_schemas"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_mcp/tool_schemas.py
# --------------------------------------------------------------------------------
