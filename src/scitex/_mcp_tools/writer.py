#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/writer.py
"""Writer module tools - delegates to scitex-writer package.

Provides usage instructions for shell-based compilation workflow.
"""

from __future__ import annotations


def register_writer_tools(mcp) -> None:
    """Register writer tools by delegating to scitex-writer package."""
    try:
        from scitex_writer._server import INSTRUCTIONS

        _SCITEX_WRITER_AVAILABLE = True
    except ImportError:
        _SCITEX_WRITER_AVAILABLE = False
        INSTRUCTIONS = None

    if not _SCITEX_WRITER_AVAILABLE:

        @mcp.tool()
        def writer_usage() -> str:
            """[writer] Get usage guide for SciTeX Writer (not installed)."""
            return "scitex-writer is required. Install with: pip install scitex-writer"

        return

    @mcp.tool()
    def writer_usage() -> str:
        """[writer] Get usage guide for SciTeX Writer LaTeX manuscript compilation system."""
        return INSTRUCTIONS


# EOF
