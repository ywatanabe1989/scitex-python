#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/mcp_server.py
# ----------------------------------------

"""
Unified FastMCP Server for SciTeX - Multi-Transport Support

Provides all SciTeX tools via a single MCP server with stdio, SSE, and HTTP.

Usage:
    scitex serve                          # stdio (Claude Desktop)
    scitex serve -t sse --port 8085       # SSE (remote via SSH)
    scitex serve -t http --port 8085      # HTTP (streamable)

Remote Setup:
    1. Local:  scitex serve -t sse --port 8085
    2. SSH:    ssh -R 8085:localhost:8085 remote-host
    3. Remote: {"type": "sse", "url": "http://localhost:8085/sse"}
"""

from __future__ import annotations

import json

try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None  # type: ignore

__all__ = ["mcp", "run_server", "main", "FASTMCP_AVAILABLE"]

if FASTMCP_AVAILABLE:
    mcp = FastMCP(
        name="scitex",
        instructions=(
            "SciTeX unified MCP server for scientific research automation. "
            "Modules: audio, capture, scholar, ui, plt, canvas, diagram, "
            "stats, template, writer."
        ),
    )
else:
    mcp = None


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


# Register tools from each module
if FASTMCP_AVAILABLE:
    from scitex._mcp_tools import register_all_tools

    register_all_tools(mcp)


def run_server(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8085):
    """Run the unified MCP server with transport selection."""
    if not FASTMCP_AVAILABLE:
        import sys

        print("=" * 60)
        print("Requires 'fastmcp' package: pip install fastmcp")
        print("=" * 60)
        sys.exit(1)

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        print(f"Starting scitex MCP (SSE) on {host}:{port}")
        print(f"Remote: ssh -R {port}:localhost:{port} remote-host")
        mcp.run(transport="sse", host=host, port=port)
    elif transport == "http":
        print(f"Starting scitex MCP (HTTP) on {host}:{port}")
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        raise ValueError(f"Unknown transport: {transport}")


def main():
    """Entry point for scitex-mcp command."""
    run_server(transport="stdio")


if __name__ == "__main__":
    main()

# EOF
