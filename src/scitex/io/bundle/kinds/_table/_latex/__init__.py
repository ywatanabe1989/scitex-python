#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/_latex/__init__.py

"""LaTeX export for FTS bundles.

This module provides LaTeX export capabilities for FTS figure and table bundles.

Usage:
    from scitex.io.bundle._tables._latex import export_to_latex, launch_editor

    # Export a bundle to LaTeX
    result = export_to_latex(bundle)
    print(result.latex_code)

    # Open editor for manual fixes
    launch_editor(result.latex_code, bundle=bundle)
"""

from ._editor import launch_editor
from ._export import LaTeXExportOptions, LaTeXResult, export_multiple, export_to_latex
from ._figure_exporter import FigureExportOptions, export_figure_to_latex, generate_figure_preamble
from ._stats_formatter import (
    FormattedStat,
    format_inline_stat,
    format_stat_note,
    format_stats_for_latex,
    format_stats_paragraph,
)
from ._table_exporter import (
    ColumnDef,
    TableExportOptions,
    export_table_to_latex,
    generate_table_preamble,
)
from ._utils import (
    column_spec,
    escape_latex,
    escape_latex_minimal,
    format_effect_size,
    format_number,
    format_p_value,
    format_statistic,
    format_unit,
    sanitize_label,
    significance_stars,
    wrap_math,
)
from ._validator import ErrorSeverity, LaTeXError, ValidationResult, validate_latex

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
    "format_stats_paragraph",
    "FormattedStat",
    # Validation
    "validate_latex",
    "ValidationResult",
    "LaTeXError",
    "ErrorSeverity",
    # Editor
    "launch_editor",
    # Utilities
    "escape_latex",
    "escape_latex_minimal",
    "format_unit",
    "format_number",
    "format_p_value",
    "format_statistic",
    "format_effect_size",
    "significance_stars",
    "sanitize_label",
    "wrap_math",
    "column_spec",
]

# EOF
