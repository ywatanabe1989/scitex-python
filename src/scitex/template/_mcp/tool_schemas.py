#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/template/_mcp.tool_schemas.py
# ----------------------------------------

"""
MCP Tool schemas for SciTeX Template module.

Defines available tools for project scaffolding:
- list_templates: List available project templates
- get_template_info: Get detailed information about a template
- clone_template: Create a new project from a template
"""

from __future__ import annotations

import mcp.types as types


def get_tool_schemas() -> list[types.Tool]:
    """Return list of available MCP tools for template operations."""
    return [
        # List available templates
        types.Tool(
            name="list_templates",
            description="List all available SciTeX project templates with their descriptions",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # Get template information
        types.Tool(
            name="get_template_info",
            description="Get detailed information about a specific project template",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_id": {
                        "type": "string",
                        "description": "Template identifier (research, pip_project, singularity, paper_directory)",
                        "enum": [
                            "research",
                            "pip_project",
                            "singularity",
                            "paper_directory",
                        ],
                    },
                },
                "required": ["template_id"],
            },
        ),
        # Clone template
        types.Tool(
            name="clone_template",
            description="Create a new project by cloning a template. Supports research projects, Python packages, Singularity containers, and paper directories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_id": {
                        "type": "string",
                        "description": "Template to clone (research, pip_project, singularity, paper_directory)",
                        "enum": [
                            "research",
                            "pip_project",
                            "singularity",
                            "paper_directory",
                        ],
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Name for the new project (will be used as directory name)",
                    },
                    "target_dir": {
                        "type": "string",
                        "description": "Parent directory where project will be created (default: current directory)",
                    },
                    "git_strategy": {
                        "type": "string",
                        "description": "Git initialization strategy",
                        "enum": ["child", "parent", "origin", "none"],
                        "default": "child",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Specific branch to clone (optional)",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Specific tag/release to clone (optional)",
                    },
                },
                "required": ["template_id", "project_name"],
            },
        ),
        # Get git strategies
        types.Tool(
            name="list_git_strategies",
            description="List available git initialization strategies for template cloning",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # Get code template
        types.Tool(
            name="get_code_template",
            description="Get a code template for @stx.session scripts, module usage, or config files. Returns ready-to-use Python code with usage guidelines. Core templates: session, io, config. Module usage: plt, stats, scholar, audio, capture, diagram, canvas, writer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_id": {
                        "type": "string",
                        "description": "Template identifier. Core (priority 1-3): session, io, config. Session variants: session-minimal, session-plot, session-stats, module. Module usage: plt, stats, scholar, audio, capture, diagram, canvas, writer. Use 'all' for all templates.",
                        "enum": [
                            # Core templates
                            "session",
                            "io",
                            "config",
                            # Session variants
                            "session-minimal",
                            "session-plot",
                            "session-stats",
                            "module",
                            # Module usage templates
                            "plt",
                            "stats",
                            "scholar",
                            "audio",
                            "capture",
                            "diagram",
                            "canvas",
                            "writer",
                            # All
                            "all",
                        ],
                    },
                    "filepath": {
                        "type": "string",
                        "description": "File path to include in template header (optional)",
                    },
                    "docstring": {
                        "type": "string",
                        "description": "Custom docstring for the template (optional)",
                    },
                },
                "required": ["template_id"],
            },
        ),
        # List code templates
        types.Tool(
            name="list_code_templates",
            description="List all available code templates for scripts and modules",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


__all__ = ["get_tool_schemas"]

# EOF
