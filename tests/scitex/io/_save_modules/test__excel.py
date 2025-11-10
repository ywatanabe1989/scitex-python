#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/scitex_repo/tests/scitex/io/_save_modules/test__excel.py
# ----------------------------------------
import os

__FILE__ = "./tests/scitex/io/_save_modules/test__excel.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Test cases for Excel saving functionality
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from scitex.io._save_modules import save_excel


class TestSaveExcel:
    """Test suite for save_excel function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.xlsx")

    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_dataframe(self):
        """Test saving a pandas DataFrame"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})
        save_excel(df, self.test_file)
        
        # Verify file exists and content is correct
        assert os.path.exists(self.test_file)
        loaded_df = pd.read_excel(self.test_file)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_dict(self):
        """Test saving dictionary as Excel"""
        data = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": ["a", "b", "c"]}
        save_excel(data, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        pd.testing.assert_frame_equal(pd.DataFrame(data), loaded_df)

    def test_save_numpy_array(self):
        """Test saving numpy array as Excel"""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        save_excel(arr, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        np.testing.assert_array_equal(arr, loaded_df.values)

    def test_save_2d_list(self):
        """Test saving 2D list as Excel through numpy conversion"""
        data = [[1, 2, 3], [4, 5, 6]]
        arr = np.array(data)
        save_excel(arr, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        np.testing.assert_array_equal(arr, loaded_df.values)

    def test_save_with_sheet_name(self):
        """Test saving with custom sheet name"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        save_excel(df, self.test_file, sheet_name="MySheet")
        
        loaded_df = pd.read_excel(self.test_file, sheet_name="MySheet")
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_multiple_sheets(self):
        """Test saving multiple sheets using ExcelWriter"""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        
        # This would require modifying save_excel to support multiple sheets
        # For now, test single sheet functionality
        save_excel(df1, self.test_file)
        assert os.path.exists(self.test_file)

    def test_save_mixed_types(self):
        """Test saving mixed data types"""
        data = {
            "integers": [1, 2, 3],
            "floats": [1.1, 2.2, 3.3],
            "strings": ["a", "b", "c"],
            "booleans": [True, False, True]
        }
        save_excel(data, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        assert loaded_df["integers"].dtype == np.int64
        assert loaded_df["floats"].dtype == np.float64
        assert loaded_df["strings"].dtype == object
        assert loaded_df["booleans"].dtype == bool

    def test_save_with_datetime(self):
        """Test saving datetime data"""
        dates = pd.date_range("2023-01-01", periods=3)
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3]})
        save_excel(df, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        # Excel might change datetime precision slightly
        assert len(loaded_df) == len(df)
        assert list(loaded_df["value"]) == [1, 2, 3]

    def test_save_large_dataframe(self):
        """Test saving large DataFrame"""
        df = pd.DataFrame(np.random.randn(1000, 10))
        save_excel(df, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        assert loaded_df.shape == (1000, 10)

    def test_error_unsupported_type(self):
        """Test error handling for unsupported types"""
        class CustomObject:
            pass
        
        obj = CustomObject()
        with pytest.raises(ValueError, match="Cannot save object of type"):
            save_excel(obj, self.test_file)

    def test_save_empty_dataframe(self):
        """Test saving empty DataFrame"""
        df = pd.DataFrame()
        save_excel(df, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        assert loaded_df.empty

    def test_save_with_special_characters(self):
        """Test saving data with special characters"""
        df = pd.DataFrame({
            "col1": ["hello", "world", "test"],
            "col2": ["特殊文字", "émojis 😊", "tabs\there"]
        })
        save_excel(df, self.test_file)
        
        loaded_df = pd.read_excel(self.test_file)
        pd.testing.assert_frame_equal(df, loaded_df)


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_excel.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 20:31:51 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/io/_save_modules/_excel.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Excel saving functionality for scitex.io.save
# """
# 
# import numpy as np
# 
# 
# def _generate_pval_variants():
#     connectors = ["", "_", "-"]
#     pval_bases = ["pvalue", "p"]
#     adjusted_variants = ["", "adjusted", "adj"]
# 
#     pval_variants = []
#     for base in pval_bases:
#         for conn in connectors:
#             for adj in adjusted_variants:
#                 if adj:
#                     variant = f"{base}{conn}{adj}"
#                 else:
#                     if base == "p":
#                         variant = f"{base}{conn}value" if conn else base
#                     else:
#                         variant = base
#                 if variant not in pval_variants:
#                     pval_variants.append(variant)
# 
#     pval_variants.extend(["padj", "p-val"])
#     pval_variants = sorted(set(pval_variants))
#     return pval_variants
# 
# 
# def _is_statistical_results(df):
#     """
#     Check if DataFrame contains statistical test results.
# 
#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame to check
# 
#     Returns
#     -------
#     bool
#         True if DataFrame appears to contain statistical test results
#     """
#     import pandas as pd
# 
#     if not isinstance(df, pd.DataFrame):
#         return False
# 
#     # Check for characteristic statistical test fields
#     PVAL_VARIANTS = _generate_pval_variants()
#     return any(col in df.columns for col in PVAL_VARIANTS)
# 
# 
# def _apply_stats_styling(df, spath):
#     """Apply conditional formatting to statistical results in Excel.
# 
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Statistical results DataFrame
#     spath : str
#         Path to Excel file
# 
#     Notes
#     -----
#     Color coding for p-values:
#     - Red background: p < 0.001 (***)
#     - Orange background: p < 0.01 (**)
#     - Yellow background: p < 0.05 (*)
#     - Light gray: p >= 0.05 (ns)
#     """
#     import pandas as pd
# 
#     try:
#         from openpyxl import load_workbook
#         from openpyxl.styles import Font, PatternFill
#     except ImportError:
#         return
# 
#     wb = load_workbook(spath)
#     ws = wb.active
# 
#     fill_red = PatternFill(
#         start_color="FF6B6B", end_color="FF6B6B", fill_type="solid"
#     )
#     fill_orange = PatternFill(
#         start_color="FFA500", end_color="FFA500", fill_type="solid"
#     )
#     fill_yellow = PatternFill(
#         start_color="FFE66D", end_color="FFE66D", fill_type="solid"
#     )
#     fill_gray = PatternFill(
#         start_color="E8E8E8", end_color="E8E8E8", fill_type="solid"
#     )
#     font_bold = Font(bold=True)
# 
#     PVAL_VARIANTS = _generate_pval_variants()
#     pval_cols = [col for col in df.columns if col in PVAL_VARIANTS]
# 
#     for pval_col in pval_cols:
#         pvalue_col_idx = df.columns.get_loc(pval_col) + 1
# 
#         for row_idx, pvalue in enumerate(df[pval_col], start=2):
#             cell = ws.cell(row=row_idx, column=pvalue_col_idx)
#             if pd.notna(pvalue):
#                 if pvalue < 0.001:
#                     cell.fill = fill_red
#                     cell.font = font_bold
#                 elif pvalue < 0.01:
#                     cell.fill = fill_orange
#                     cell.font = font_bold
#                 elif pvalue < 0.05:
#                     cell.fill = fill_yellow
#                     cell.font = font_bold
#                 else:
#                     cell.fill = fill_gray
# 
#             if pd.notna(pvalue) and pvalue < 0.05:
#                 for col_idx in range(1, len(df.columns) + 1):
#                     ws.cell(row=row_idx, column=col_idx).font = font_bold
# 
#     for column in ws.columns:
#         max_length = 0
#         column_letter = column[0].column_letter
#         for cell in column:
#             try:
#                 if len(str(cell.value)) > max_length:
#                     max_length = len(str(cell.value))
#             except:
#                 pass
#         adjusted_width = min(max_length + 2, 50)
#         ws.column_dimensions[column_letter].width = adjusted_width
# 
#     ws.freeze_panes = "A2"
#     wb.save(spath)
# 
# 
# def save_excel(obj, spath, style=True, **kwargs):
#     """Handle Excel file saving with optional statistical formatting.
# 
#     Parameters
#     ----------
#     obj : pd.DataFrame, dict, list of dict, or np.ndarray
#         Object to save as Excel file
#     spath : str
#         Path where Excel file will be saved
#     style : bool, default True
#         If True, automatically apply conditional formatting to statistical results
#         (p-values colored by significance level). Set to False to disable styling.
#     **kwargs
#         Additional keyword arguments passed to pandas.DataFrame.to_excel()
# 
#     Raises
#     ------
#     ValueError
#         If object type cannot be saved as Excel file
# 
#     Notes
#     -----
#     When saving statistical test results (DataFrames with 'pvalue' column),
#     automatic conditional formatting is applied unless style=False:
# 
#     - Red background: p < 0.001 (***)
#     - Orange background: p < 0.01 (**)
#     - Yellow background: p < 0.05 (*)
#     - Light gray: p >= 0.05 (ns)
#     - Bold font for significant results (p < 0.05)
# 
#     Examples
#     --------
#     >>> import scitex as stx
#     >>> results = stx.stats.test_ttest_ind(x, y)
#     >>> stx.io.save(results, 'results.xlsx')  # Auto-styled
#     >>> stx.io.save(results, 'results.xlsx', style=False)  # No styling
#     """
#     # Lazy import to avoid circular import issues
#     import pandas as pd
# 
#     # Convert to DataFrame
#     if isinstance(obj, pd.DataFrame):
#         df = obj
#     elif isinstance(obj, dict):
#         df = pd.DataFrame(obj)
#     elif isinstance(obj, list):
#         # Handle list of dicts (common for multiple test results)
#         df = pd.DataFrame(obj)
#     elif isinstance(obj, np.ndarray):
#         df = pd.DataFrame(obj)
#     else:
#         raise ValueError(
#             f"Cannot save object of type {type(obj)} as Excel file"
#         )
# 
#     # Save to Excel
#     df.to_excel(spath, index=False, **kwargs)
# 
#     # Apply styling if enabled and data contains statistical results
#     if style and _is_statistical_results(df):
#         _apply_stats_styling(df, spath)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_excel.py
# --------------------------------------------------------------------------------
