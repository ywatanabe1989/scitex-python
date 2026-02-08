#!/usr/bin/env python3
"""SciTeX Linter — thin wrapper delegating to scitex-linter package.

Usage:
    import scitex as stx
    issues = stx.linter.lint_file("script.py")
"""

import os as _os

# Set branding BEFORE importing scitex-linter
_os.environ.setdefault("SCITEX_LINTER_BRAND", "scitex.linter")
_os.environ.setdefault("SCITEX_LINTER_ALIAS", "linter")

try:
    from scitex_linter.checker import lint_file, lint_source
    from scitex_linter.formatter import format_issue, format_summary, to_json
    from scitex_linter.rules import ALL_RULES
except ImportError:

    def lint_file(*args, **kwargs):
        raise ImportError(
            "scitex-linter is required. Install with: pip install scitex-linter"
        )

    def lint_source(*args, **kwargs):
        raise ImportError(
            "scitex-linter is required. Install with: pip install scitex-linter"
        )

    format_issue = None
    format_summary = None
    to_json = None
    ALL_RULES = {}

__all__ = [
    "lint_file",
    "lint_source",
    "format_issue",
    "format_summary",
    "to_json",
    "ALL_RULES",
]

# EOF
