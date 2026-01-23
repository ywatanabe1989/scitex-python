#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: src/scitex/writer/_mcp/tool_schemas.py

"""
MCP Tool schemas for SciTeX Writer module.

Provides usage instructions for shell-based compilation workflow.
"""

from __future__ import annotations

import mcp.types as types


def get_tool_schemas() -> list[types.Tool]:
    """Return list of available MCP tools for writer operations."""
    return [
        types.Tool(
            name="writer_usage",
            description="[writer] Get usage guide for SciTeX Writer LaTeX manuscript compilation system.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


__all__ = ["get_tool_schemas"]

# EOF
