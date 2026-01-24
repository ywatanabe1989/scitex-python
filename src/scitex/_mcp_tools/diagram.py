#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/diagram.py
"""Diagram module tools for FastMCP unified server.

Delegates entirely to figrecipe's diagram MCP tools.
figrecipe is the single source of truth.
"""

from __future__ import annotations


def register_diagram_tools(mcp) -> None:
    """Register diagram tools by delegating to figrecipe."""
    from figrecipe._mcp._diagram_tools import (
        register_diagram_tools as register_figrecipe_diagram_tools,
    )

    register_figrecipe_diagram_tools(mcp)


# EOF
