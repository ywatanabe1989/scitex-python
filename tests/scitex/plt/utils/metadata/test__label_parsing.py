# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_label_parsing.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_label_parsing.py
# 
# """
# Label and unit parsing utilities.
# 
# This module provides functions to parse axis labels and extract unit information
# from formatted strings like "Time [s]" or "Amplitude (a.u.)".
# """
# 
# import re
# 
# 
# def _parse_label_unit(label_text: str) -> tuple:
#     """
#     Parse label text to extract label and unit.
# 
#     Handles formats like:
#     - "Time [s]" -> ("Time", "s")
#     - "Amplitude (a.u.)" -> ("Amplitude", "a.u.")
#     - "Value" -> ("Value", "")
# 
#     Parameters
#     ----------
#     label_text : str
#         The full label text from axes
# 
#     Returns
#     -------
#     tuple
#         (label, unit) where unit is empty string if not found
#     """
#     if not label_text:
#         return "", ""
# 
#     # Try to match [...] pattern first (preferred format)
#     match = re.match(r"^(.+?)\s*\[([^\]]+)\]$", label_text)
#     if match:
#         return match.group(1).strip(), match.group(2).strip()
# 
#     # Try to match (...) pattern
#     match = re.match(r"^(.+?)\s*\(([^\)]+)\)$", label_text)
#     if match:
#         return match.group(1).strip(), match.group(2).strip()
# 
#     # No unit found
#     return label_text.strip(), ""
# 
# 
# def _get_csv_column_names(trace_id: str, ax_row: int = 0, ax_col: int = 0, variables: list = None) -> dict:
#     """
#     Get CSV column names using the single source of truth naming convention.
# 
#     Format: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
# 
#     Parameters
#     ----------
#     trace_id : str
#         The trace identifier (e.g., "sine", "step")
#     ax_row : int
#         Row position of axes in grid (default: 0)
#     ax_col : int
#         Column position of axes in grid (default: 0)
#     variables : list, optional
#         List of variable names (default: ["x", "y"])
# 
#     Returns
#     -------
#     dict
#         Dictionary mapping variable names to CSV column names
#     """
#     from .._csv_column_naming import get_csv_column_name
# 
#     if variables is None:
#         variables = ["x", "y"]
# 
#     data_ref = {}
#     for var in variables:
#         data_ref[var] = get_csv_column_name(var, ax_row, ax_col, trace_id=trace_id)
# 
#     return data_ref

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_label_parsing.py
# --------------------------------------------------------------------------------
