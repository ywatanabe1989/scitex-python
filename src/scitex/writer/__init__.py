#!/usr/bin/env python3
"""
SciTeX Writer - Thin wrapper delegating to scitex-writer package.

Single source of truth: scitex-writer package
This module re-exports scitex-writer for convenience.

Install: pip install scitex-writer
"""

import os as _os

# =============================================================================
# Set branding environment variables BEFORE importing scitex-writer
# (Will take effect once scitex-writer implements branding support)
# =============================================================================
_os.environ.setdefault("SCITEX_WRITER_BRAND", "scitex.writer")
_os.environ.setdefault("SCITEX_WRITER_ALIAS", "sw")

# =============================================================================
# Re-export from scitex-writer package (single source of truth)
# =============================================================================
try:
    from scitex_writer import __version__ as writer_version
    from scitex_writer import (
        bib,
        build_guideline,
        compile,
        figures,
        generate_ai2_prompt,
        generate_asta,
        get_guideline,
        guidelines,
        list_guidelines,
        project,
        prompts,
        tables,
    )

    HAS_WRITER_PKG = True

except ImportError:
    HAS_WRITER_PKG = False
    writer_version = None
    bib = None
    compile = None
    figures = None
    guidelines = None
    project = None
    prompts = None
    tables = None
    get_guideline = None
    build_guideline = None
    list_guidelines = None
    generate_ai2_prompt = None
    generate_asta = None


def _check_writer_pkg():
    """Check if scitex-writer package is available."""
    if not HAS_WRITER_PKG:
        raise ImportError(
            "scitex-writer package not installed. "
            "Install with: pip install scitex-writer"
        )


__all__ = [
    # Package availability
    "HAS_WRITER_PKG",
    "writer_version",
    # Modules
    "compile",
    "project",
    "tables",
    "figures",
    "bib",
    "guidelines",
    "prompts",
    # Convenience functions
    "get_guideline",
    "build_guideline",
    "list_guidelines",
    "generate_ai2_prompt",
    "generate_asta",
]

# EOF
