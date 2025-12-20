#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/__init__.py

"""Table handling and LaTeX export for FTS bundles.

This module provides:
- LaTeX export for figures and tables (booktabs style)
- Statistical result formatting
- LaTeX validation and error editor
"""

from ._latex import (
    ColumnDef,
    ErrorSeverity,
    FigureExportOptions,
    FormattedStat,
    LaTeXError,
    LaTeXExportOptions,
    LaTeXResult,
    TableExportOptions,
    ValidationResult,
    export_figure_to_latex,
    export_multiple,
    export_table_to_latex,
    export_to_latex,
    format_inline_stat,
    format_stat_note,
    format_stats_for_latex,
    generate_figure_preamble,
    generate_table_preamble,
    launch_editor,
    validate_latex,
)

__all__ = [
    # Main export API
    "export_to_latex",
    "export_multiple",
    "LaTeXExportOptions",
    "LaTeXResult",
    # Figure export
    "export_figure_to_latex",
    "FigureExportOptions",
    "generate_figure_preamble",
    # Table export
    "export_table_to_latex",
    "TableExportOptions",
    "ColumnDef",
    "generate_table_preamble",
    # Stats formatting
    "format_stats_for_latex",
    "format_inline_stat",
    "format_stat_note",
    "FormattedStat",
    # Validation
    "validate_latex",
    "ValidationResult",
    "LaTeXError",
    "ErrorSeverity",
    # Editor
    "launch_editor",
]

# EOF
