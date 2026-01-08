#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/diagram/_mcp_tool_schemas.py
# ----------------------------------------

"""
MCP Tool schemas for SciTeX diagram module.

Defines tools for paper-optimized diagram generation:
- create_diagram: Create diagram from YAML spec
- compile_mermaid: Export to Mermaid format
- compile_graphviz: Export to Graphviz DOT format
- list_presets: List available diagram presets
- apply_preset: Apply workflow/decision/pipeline preset
"""

from __future__ import annotations

import mcp.types as types


def get_tool_schemas() -> list[types.Tool]:
    """Return list of available MCP tools for diagram operations."""
    return [
        # Create diagram from YAML
        types.Tool(
            name="create_diagram",
            description="Create a diagram from a YAML specification file or dictionary",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec_path": {
                        "type": "string",
                        "description": "Path to YAML specification file",
                    },
                    "spec_dict": {
                        "type": "object",
                        "description": "Diagram specification as dictionary (alternative to spec_path)",
                    },
                },
                "required": [],
            },
        ),
        # Compile to Mermaid
        types.Tool(
            name="compile_mermaid",
            description="Compile diagram specification to Mermaid format",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec_path": {
                        "type": "string",
                        "description": "Path to YAML specification file",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path for .mmd file",
                    },
                    "spec_dict": {
                        "type": "object",
                        "description": "Diagram specification as dictionary (alternative to spec_path)",
                    },
                },
                "required": [],
            },
        ),
        # Compile to Graphviz
        types.Tool(
            name="compile_graphviz",
            description="Compile diagram specification to Graphviz DOT format",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec_path": {
                        "type": "string",
                        "description": "Path to YAML specification file",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path for .dot file",
                    },
                    "spec_dict": {
                        "type": "object",
                        "description": "Diagram specification as dictionary (alternative to spec_path)",
                    },
                },
                "required": [],
            },
        ),
        # List presets
        types.Tool(
            name="list_presets",
            description="List available diagram presets (workflow, decision, pipeline)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # Get preset
        types.Tool(
            name="get_preset",
            description="Get a diagram preset configuration by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "preset_name": {
                        "type": "string",
                        "description": "Preset name",
                        "enum": ["workflow", "decision", "pipeline"],
                    },
                },
                "required": ["preset_name"],
            },
        ),
        # Split diagram
        types.Tool(
            name="split_diagram",
            description="Split a large diagram into smaller parts for multi-column layouts",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec_path": {
                        "type": "string",
                        "description": "Path to YAML specification file",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Split strategy",
                        "enum": ["horizontal", "vertical", "semantic"],
                        "default": "horizontal",
                    },
                    "max_nodes_per_part": {
                        "type": "integer",
                        "description": "Maximum nodes per split part",
                        "default": 10,
                    },
                },
                "required": ["spec_path"],
            },
        ),
        # Get paper modes
        types.Tool(
            name="get_paper_modes",
            description="Get available paper layout modes and their constraints",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


__all__ = ["get_tool_schemas"]

# EOF
