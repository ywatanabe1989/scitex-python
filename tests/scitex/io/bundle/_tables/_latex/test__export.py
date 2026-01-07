# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_export.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/_latex/_export.py
# 
# """Main LaTeX export orchestration for FTS bundles."""
# 
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import TYPE_CHECKING, List, Optional, Union
# 
# from ._figure_exporter import FigureExportOptions, export_figure_to_latex, generate_figure_preamble
# from ._stats_formatter import FormattedStat, format_stat_note, format_stats_for_latex
# from ._table_exporter import TableExportOptions, export_table_to_latex, generate_table_preamble
# from ._validator import LaTeXError, ValidationResult, validate_latex
# 
# if TYPE_CHECKING:
#     from scitex.io.bundle import FTS
# 
# 
# @dataclass
# class LaTeXExportOptions:
#     """Options for LaTeX export.
# 
#     Attributes:
#         validate: Whether to validate output
#         validation_level: Validation level ("syntax", "semantic", "compile")
#         include_preamble: Include required packages in output
#         figure_options: Options for figure export
#         table_options: Options for table export
#         output_path: Where to save the .tex file
#     """
# 
#     validate: bool = True
#     validation_level: str = "syntax"
#     include_preamble: bool = False
#     figure_options: Optional[FigureExportOptions] = None
#     table_options: Optional[TableExportOptions] = None
#     output_path: Optional[Path] = None
# 
# 
# @dataclass
# class LaTeXResult:
#     """Result of LaTeX export.
# 
#     Attributes:
#         latex_code: Generated LaTeX code
#         preamble: Required preamble packages
#         validation: Validation result
#         output_path: Path where file was saved (if applicable)
#         stats: Formatted statistics (if applicable)
#     """
# 
#     latex_code: str
#     preamble: str = ""
#     validation: Optional[ValidationResult] = None
#     output_path: Optional[Path] = None
#     stats: List[FormattedStat] = field(default_factory=list)
# 
#     @property
#     def is_valid(self) -> bool:
#         """Check if export is valid."""
#         return self.validation is None or self.validation.is_valid
# 
#     @property
#     def errors(self) -> List[LaTeXError]:
#         """Get validation errors."""
#         if self.validation:
#             return self.validation.errors
#         return []
# 
#     @property
#     def warnings(self) -> List[LaTeXError]:
#         """Get validation warnings."""
#         if self.validation:
#             return self.validation.warnings
#         return []
# 
# 
# def export_to_latex(
#     bundle: "FTS",
#     options: Optional[LaTeXExportOptions] = None,
# ) -> LaTeXResult:
#     """Export an FTS bundle to LaTeX.
# 
#     Automatically detects bundle type and uses appropriate exporter.
# 
#     Args:
#         bundle: FTS bundle
#         options: Export options
# 
#     Returns:
#         LaTeXResult with code, validation, and metadata
#     """
#     if options is None:
#         options = LaTeXExportOptions()
# 
#     node_type = bundle.node.type
#     latex_code = ""
#     preamble_parts = []
#     stats = []
# 
#     # Export based on type
#     if node_type in ("figure", "plot", "image"):
#         latex_code = export_figure_to_latex(
#             bundle,
#             options=options.figure_options,
#         )
#         preamble_parts.append(generate_figure_preamble())
# 
#     elif node_type == "table":
#         table_opts = options.table_options or TableExportOptions()
#         latex_code = export_table_to_latex(
#             bundle,
#             options=table_opts,
#         )
#         preamble_parts.append(
#             generate_table_preamble(
#                 use_booktabs=table_opts.use_booktabs,
#                 use_siunitx=table_opts.use_siunitx,
#             )
#         )
# 
#     elif node_type == "stats":
#         # Stats-only bundle - format as paragraph or inline
#         if bundle.stats:
#             stats = format_stats_for_latex(bundle.stats.to_dict())
#             latex_code = _format_stats_only(stats)
# 
#     else:
#         raise ValueError(f"Unsupported node type for LaTeX export: {node_type}")
# 
#     # Format statistics if available (for figures/tables with stats)
#     if bundle.stats and node_type != "stats":
#         stats = format_stats_for_latex(bundle.stats.to_dict())
#         if stats:
#             note = format_stat_note(stats)
#             if note:
#                 latex_code += "\n\n" + note
# 
#     # Combine preamble
#     preamble = "\n\n".join(preamble_parts)
# 
#     # Validate if requested
#     validation = None
#     if options.validate:
#         validation = validate_latex(latex_code, level=options.validation_level)
# 
#     # Create result
#     result = LaTeXResult(
#         latex_code=latex_code,
#         preamble=preamble,
#         validation=validation,
#         stats=stats,
#     )
# 
#     # Save to file if path specified
#     output_path = options.output_path
#     if output_path is None:
#         # Default to exports directory
#         exports_dir = bundle.path / "exports"
#         exports_dir.mkdir(exist_ok=True)
#         output_path = exports_dir / f"{bundle.node.id}.tex"
# 
#     if output_path:
#         _save_latex(latex_code, output_path, preamble if options.include_preamble else None)
#         result.output_path = output_path
# 
#     return result
# 
# 
# def _format_stats_only(stats: List[FormattedStat]) -> str:
#     """Format stats-only bundle as LaTeX.
# 
#     Args:
#         stats: Formatted statistics
# 
#     Returns:
#         LaTeX code
#     """
#     if not stats:
#         return "% No statistics to report"
# 
#     lines = ["% Statistical Results", ""]
#     for stat in stats:
#         lines.append(f"% {stat.test_type}")
#         lines.append(stat.full_report)
#         lines.append("")
# 
#     return "\n".join(lines)
# 
# 
# def _save_latex(
#     latex_code: str,
#     output_path: Path,
#     preamble: Optional[str] = None,
# ) -> None:
#     """Save LaTeX code to file.
# 
#     Args:
#         latex_code: LaTeX code
#         output_path: Output path
#         preamble: Optional preamble to prepend
#     """
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
# 
#     content = []
#     if preamble:
#         content.append(preamble)
#         content.append("")
# 
#     content.append(latex_code)
# 
#     output_path.write_text("\n".join(content), encoding="utf-8")
# 
# 
# def export_multiple(
#     bundles: List["FTS"],
#     output_path: Path,
#     options: Optional[LaTeXExportOptions] = None,
# ) -> LaTeXResult:
#     """Export multiple bundles to a single LaTeX file.
# 
#     Args:
#         bundles: List of FTS bundles
#         output_path: Output path
#         options: Export options
# 
#     Returns:
#         Combined LaTeXResult
#     """
#     if options is None:
#         options = LaTeXExportOptions()
# 
#     all_code = []
#     all_preamble = set()
#     all_stats = []
#     all_errors = []
#     all_warnings = []
# 
#     for bundle in bundles:
#         result = export_to_latex(bundle, options)
#         all_code.append(result.latex_code)
#         if result.preamble:
#             all_preamble.add(result.preamble)
#         all_stats.extend(result.stats)
#         all_errors.extend(result.errors)
#         all_warnings.extend(result.warnings)
# 
#     combined_code = "\n\n".join(all_code)
#     combined_preamble = "\n\n".join(sorted(all_preamble))
# 
#     # Save combined file
#     _save_latex(combined_code, output_path, combined_preamble if options.include_preamble else None)
# 
#     # Create combined validation result
#     combined_validation = ValidationResult()
#     for error in all_errors:
#         combined_validation.add_error(error)
#     for warning in all_warnings:
#         combined_validation.add_error(warning)
# 
#     return LaTeXResult(
#         latex_code=combined_code,
#         preamble=combined_preamble,
#         validation=combined_validation,
#         output_path=output_path,
#         stats=all_stats,
#     )
# 
# 
# __all__ = [
#     "LaTeXExportOptions",
#     "LaTeXResult",
#     "export_to_latex",
#     "export_multiple",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_export.py
# --------------------------------------------------------------------------------
