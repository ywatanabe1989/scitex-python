#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: src/scitex/diagram/mcp_server.py

"""
MCP Server for SciTeX diagram - delegates to figrecipe.

This is a thin wrapper that redirects to figrecipe's MCP server.
figrecipe is the single source of truth for diagram functionality.
"""

from __future__ import annotations


def main():
    """Main entry point - delegates to figrecipe's MCP server."""
    try:
        from figrecipe._mcp.server import mcp

        mcp.run()
    except ImportError as e:
        import sys

        print("=" * 60)
        print("scitex-diagram requires figrecipe with MCP support.")
        print()
        print("Install with:")
        print("  pip install figrecipe[mcp]")
        print()
        print(f"Error: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()


# EOF
