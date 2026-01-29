#!/usr/bin/env python3
"""
SciTeX Writer - Thin wrapper delegating to scitex-writer package.

Single source of truth: scitex-writer package
This module re-exports scitex-writer as-is, without modifications.

Install: pip install scitex-writer
"""

import os as _os

# Set branding environment variables BEFORE importing scitex-writer
_os.environ.setdefault("SCITEX_WRITER_BRAND", "scitex.writer")
_os.environ.setdefault("SCITEX_WRITER_ALIAS", "sw")

# Re-export from scitex-writer package (single source of truth)
try:
    from scitex_writer import __version__ as writer_version
    from scitex_writer import (
        bib,
        compile,
        figures,
        guidelines,
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

__all__ = [
    "HAS_WRITER_PKG",
    "writer_version",
    "bib",
    "compile",
    "figures",
    "guidelines",
    "project",
    "prompts",
    "tables",
]

# EOF
