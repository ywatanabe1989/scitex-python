#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_resources/__init__.py
"""MCP Resources for SciTeX - Dynamic documentation for AI agents.

This module provides MCP resources that expose SciTeX documentation,
patterns, and examples to AI agents for code generation guidance.

Resources:
- scitex://cheatsheet - Quick reference for all core patterns
- scitex://session-tree - Output directory structure explanation
- scitex://module/{name} - Module-specific documentation (io, plt, stats, scholar, session)
- scitex://io-formats - Supported file formats
- scitex://figrecipe-spec - Figure recipe specification
"""

from __future__ import annotations

from ._cheatsheet import register_cheatsheet_resources
from ._figrecipe import register_figrecipe_resources
from ._formats import register_format_resources
from ._modules import register_module_resources
from ._scholar import register_scholar_resources
from ._session import register_session_resources

__all__ = ["register_resources"]


def register_resources(mcp) -> None:
    """Register all MCP resources with the FastMCP server."""
    register_cheatsheet_resources(mcp)
    register_session_resources(mcp)
    register_module_resources(mcp)
    register_format_resources(mcp)
    register_figrecipe_resources(mcp)
    register_scholar_resources(mcp)


# EOF
