#!/usr/bin/env python3
# Timestamp: 2026-01-27
# File: /home/ywatanabe/proj/scitex-python/src/scitex/_mcp_tools/social.py
"""Social module tools - thin wrapper delegating to socialia package.

Single source of truth: socialia MCP tools.
"""

from __future__ import annotations


def register_social_tools(mcp) -> None:
    """Register social tools by delegating to socialia package."""
    try:
        from socialia._mcp.tools import register_all_tools

        # Delegate all MCP tools to socialia
        register_all_tools(mcp)
    except ImportError:
        # Fallback when socialia is not installed
        @mcp.tool()
        def social_status() -> str:
            """[social] Get social media status (not installed)."""
            return "socialia is required. Install with: pip install socialia"


# EOF
