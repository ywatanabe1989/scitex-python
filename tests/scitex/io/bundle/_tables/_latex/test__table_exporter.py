# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_table_exporter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/_latex/_table_exporter.py
# 
# """Export FTS tables to LaTeX using booktabs."""
# 
# import csv
# import json
# from dataclasses import dataclass, field
# from io import StringIO
# from pathlib import Path
# from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
# 
# from ._utils import (
#     column_spec,
#     escape_latex,
#     escape_latex_minimal,
#     format_number,
#     format_unit,
#     sanitize_label,
# )
# 
# if TYPE_CHECKING:
#     from scitex.io.bundle import FTS
# 
# 
# # Column role to header mapping
# ROLE_HEADERS = {
#     "variable": "Variable",
#     "estimate": "Mean",
#     "dispersion": "SD",
#     "sample_size": "n",
#     "se": "SE",
#     "ci_lower": "95\\% CI lower",
#     "ci_upper": "95\\% CI upper",
#     "p_value": "$p$",
#     "statistic": "Statistic",
#     "effect_size": "Effect size",
#     "category": "Category",
#     "count": "Count",
#     "percentage": "\\%",
#     "median": "Median",
#     "iqr": "IQR",
#     "min": "Min",
#     "max": "Max",
#     "range": "Range",
# }
# 
# 
# @dataclass
# class ColumnDef:
#     """Column definition for table export.
# 
#     Attributes:
#         name: Column name in data
#         role: Semantic role (estimate, dispersion, etc.)
#         header: Display header (auto-generated from role if not specified)
#         unit: Unit string (e.g., "ms", "Hz")
#         precision: Decimal places for numeric values
#         alignment: Column alignment (l, c, r)
#     """
# 
#     name: str
#     role: Optional[str] = None
#     header: Optional[str] = None
#     unit: Optional[str] = None
#     precision: int = 2
#     alignment: str = "r"
# 
#     def get_header(self) -> str:
#         """Get display header with unit if applicable."""
#         if self.header:
#             base = self.header
#         elif self.role:
#             base = ROLE_HEADERS.get(self.role, self.role.capitalize())
#         else:
#             base = escape_latex(self.name)
# 
#         if self.unit:
#             return f"{base} ({escape_latex(self.unit)})"
#         return base
# 
# 
# @dataclass
# class TableExportOptions:
#     """Options for table LaTeX export.
# 
#     Attributes:
#         placement: Float placement specifier
#         centering: Center the table
#         include_caption: Include caption from node.name
#         include_label: Include label from node.id
#         use_booktabs: Use booktabs rules (recommended)
#         use_siunitx: Use siunitx for number formatting
#         column_defs: Override column definitions
#         precision: Default decimal precision
#         caption_position: 'top' or 'bottom'
#         fontsize: Table font size (small, footnotesize, scriptsize, etc.)
#     """
# 
#     placement: str = "htbp"
#     centering: bool = True
#     include_caption: bool = True
#     include_label: bool = True
#     use_booktabs: bool = True
#     use_siunitx: bool = False
#     column_defs: Optional[List[ColumnDef]] = None
#     precision: int = 2
#     caption_position: str = "top"
#     fontsize: Optional[str] = None
# 
# 
# def export_table_to_latex(
#     bundle: "FTS",
#     options: Optional[TableExportOptions] = None,
# ) -> str:
#     """Export a table bundle to LaTeX.
# 
#     Args:
#         bundle: FTS bundle of type 'table'
#         options: Export options
# 
#     Returns:
#         LaTeX code string
#     """
#     if options is None:
#         options = TableExportOptions()
# 
#     node = bundle.node
# 
#     # Validate type
#     if node.type != "table":
#         raise ValueError(f"Cannot export node type '{node.type}' as table. Expected 'table'.")
# 
#     # Get column definitions
#     col_defs = options.column_defs or _extract_column_defs(bundle, options)
# 
#     # Get data
#     data_rows = _load_table_data(bundle)
# 
#     # Get caption and label
#     caption = node.name or ""
#     label = sanitize_label(node.id)
# 
#     # Build LaTeX
#     lines = []
#     lines.append(f"\\begin{{table}}[{options.placement}]")
# 
#     if options.centering:
#         lines.append("    \\centering")
# 
#     if options.fontsize:
#         lines.append(f"    \\{options.fontsize}")
# 
#     # Caption at top
#     if options.include_caption and caption and options.caption_position == "top":
#         escaped_caption = escape_latex_minimal(caption)
#         lines.append(f"    \\caption{{{escaped_caption}}}")
# 
#     if options.include_label:
#         lines.append(f"    \\label{{tab:{label}}}")
# 
#     # Column spec
#     num_cols = len(col_defs)
#     alignments = "".join(col.alignment for col in col_defs)
# 
#     lines.append(f"    \\begin{{tabular}}{{{alignments}}}")
# 
#     # Header
#     if options.use_booktabs:
#         lines.append("        \\toprule")
#     else:
#         lines.append("        \\hline")
# 
#     headers = [col.get_header() for col in col_defs]
#     lines.append("        " + " & ".join(headers) + " \\\\")
# 
#     if options.use_booktabs:
#         lines.append("        \\midrule")
#     else:
#         lines.append("        \\hline")
# 
#     # Data rows
#     for row in data_rows:
#         formatted_row = []
#         for col in col_defs:
#             value = row.get(col.name, "")
#             formatted_value = _format_cell(value, col, options)
#             formatted_row.append(formatted_value)
#         lines.append("        " + " & ".join(formatted_row) + " \\\\")
# 
#     # Footer
#     if options.use_booktabs:
#         lines.append("        \\bottomrule")
#     else:
#         lines.append("        \\hline")
# 
#     lines.append("    \\end{tabular}")
# 
#     # Caption at bottom
#     if options.include_caption and caption and options.caption_position == "bottom":
#         escaped_caption = escape_latex_minimal(caption)
#         lines.append(f"    \\caption{{{escaped_caption}}}")
# 
#     lines.append("\\end{table}")
# 
#     return "\n".join(lines)
# 
# 
# def _extract_column_defs(bundle: "FTS", options: TableExportOptions) -> List[ColumnDef]:
#     """Extract column definitions from bundle encoding and data_info.
# 
#     Args:
#         bundle: FTS bundle
#         options: Export options
# 
#     Returns:
#         List of ColumnDef objects
#     """
#     col_defs = []
# 
#     # Try to get from encoding.json
#     encoding = bundle.encoding_dict
#     if encoding and "columns" in encoding:
#         for col_enc in encoding["columns"]:
#             col_def = ColumnDef(
#                 name=col_enc.get("name", ""),
#                 role=col_enc.get("role"),
#                 header=col_enc.get("header"),
#                 unit=col_enc.get("unit"),
#                 precision=col_enc.get("precision", options.precision),
#                 alignment=col_enc.get("alignment", "r"),
#             )
#             col_defs.append(col_def)
# 
#     # If no encoding, try data_info
#     if not col_defs:
#         data_info = bundle.data_info
#         if data_info and hasattr(data_info, "columns"):
#             for col_info in data_info.columns:
#                 col_name = col_info.name if hasattr(col_info, "name") else str(col_info)
#                 col_def = ColumnDef(
#                     name=col_name,
#                     precision=options.precision,
#                 )
#                 col_defs.append(col_def)
# 
#     # If still no columns, try to infer from data
#     if not col_defs:
#         data_rows = _load_table_data(bundle)
#         if data_rows:
#             for key in data_rows[0].keys():
#                 col_defs.append(ColumnDef(name=key, precision=options.precision))
# 
#     return col_defs
# 
# 
# def _load_table_data(bundle: "FTS") -> List[Dict[str, Any]]:
#     """Load table data from bundle.
# 
#     Args:
#         bundle: FTS bundle
# 
#     Returns:
#         List of row dictionaries
#     """
#     data_dir = bundle.path / "data"
# 
#     # Try CSV first
#     csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
#     if csv_files:
#         csv_path = csv_files[0]
#         try:
#             with open(csv_path, "r", newline="", encoding="utf-8") as f:
#                 reader = csv.DictReader(f)
#                 return list(reader)
#         except Exception:
#             pass
# 
#     # Try JSON
#     json_files = list(data_dir.glob("*.json")) if data_dir.exists() else []
#     for json_path in json_files:
#         if json_path.name == "data_info.json":
#             continue
#         try:
#             with open(json_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 if isinstance(data, list):
#                     return data
#                 elif isinstance(data, dict) and "rows" in data:
#                     return data["rows"]
#         except Exception:
#             pass
# 
#     return []
# 
# 
# def _format_cell(value: Any, col: ColumnDef, options: TableExportOptions) -> str:
#     """Format a cell value for LaTeX.
# 
#     Args:
#         value: Cell value
#         col: Column definition
#         options: Export options
# 
#     Returns:
#         Formatted cell string
#     """
#     if value is None or value == "":
#         return "---"
# 
#     # Handle numeric values
#     if isinstance(value, (int, float)):
#         return format_number(value, precision=col.precision, use_siunitx=options.use_siunitx)
# 
#     # Handle string numeric values
#     try:
#         num_val = float(value)
#         return format_number(num_val, precision=col.precision, use_siunitx=options.use_siunitx)
#     except (ValueError, TypeError):
#         pass
# 
#     # Text value
#     return escape_latex(str(value))
# 
# 
# def generate_table_preamble(use_booktabs: bool = True, use_siunitx: bool = False) -> str:
#     """Generate required LaTeX preamble for tables.
# 
#     Args:
#         use_booktabs: Include booktabs package
#         use_siunitx: Include siunitx package
# 
#     Returns:
#         Preamble code with required packages
#     """
#     packages = []
#     packages.append("% Required packages for table export")
# 
#     if use_booktabs:
#         packages.append("\\usepackage{booktabs}")
# 
#     if use_siunitx:
#         packages.append("\\usepackage{siunitx}")
#         packages.append("\\sisetup{")
#         packages.append("    round-mode = places,")
#         packages.append("    round-precision = 2,")
#         packages.append("    detect-all,")
#         packages.append("}")
# 
#     return "\n".join(packages)
# 
# 
# __all__ = [
#     "ColumnDef",
#     "TableExportOptions",
#     "export_table_to_latex",
#     "generate_table_preamble",
#     "ROLE_HEADERS",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_table_exporter.py
# --------------------------------------------------------------------------------
