#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: src/scitex/writer/_mcp/handlers.py

"""
MCP Handler for SciTeX Writer module.

Delegates to scitex-writer package for usage instructions.
"""

from __future__ import annotations


async def writer_usage_handler() -> dict:
    """Get usage guide for SciTeX Writer."""
    try:
        from scitex_writer._server import INSTRUCTIONS

        return {
            "success": True,
            "instructions": INSTRUCTIONS,
        }
    except ImportError:
        return {
            "success": False,
            "error": "scitex-writer is required. Install with: pip install scitex-writer",
        }


__all__ = ["writer_usage_handler"]

# EOF
