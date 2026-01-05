# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_csv_hash.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_csv_hash.py
# 
# """
# CSV data hash computation utilities.
# 
# This module provides functions to compute hashes of CSV data for reproducibility
# verification and linking JSON metadata to CSV files.
# """
# 
# from typing import Optional
# import hashlib
# import pandas as pd
# 
# 
# def _compute_csv_hash_from_df(df) -> Optional[str]:
#     """
#     Compute a hash of CSV data from a DataFrame.
# 
#     This is used after actual CSV export to compute the hash from the
#     exact data that was written.
# 
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The DataFrame to compute hash from
# 
#     Returns
#     -------
#     str or None
#         SHA256 hash of the CSV data (first 16 chars), or None if unable to compute
#     """
#     try:
#         if df is None or df.empty:
#             return None
# 
#         # Convert to CSV string for hashing
#         csv_string = df.to_csv(index=False)
# 
#         # Compute SHA256 hash
#         hash_obj = hashlib.sha256(csv_string.encode("utf-8"))
#         hash_hex = hash_obj.hexdigest()
# 
#         # Return first 16 characters for readability
#         return hash_hex[:16]
# 
#     except Exception:
#         return None
# 
# 
# def _compute_csv_hash(ax_or_df) -> Optional[str]:
#     """
#     Compute a hash of the CSV data for reproducibility verification.
# 
#     The hash is computed from the actual data that would be exported to CSV,
#     allowing verification that JSON and CSV files are in sync.
# 
#     Note: The hash is computed from the AxisWrapper's export_as_csv(), which
#     does NOT include the ax_{index:02d}_ prefix. The FigWrapper.export_as_csv()
#     adds this prefix. We replicate this prefix addition here.
# 
#     Parameters
#     ----------
#     ax_or_df : AxisWrapper, matplotlib.axes.Axes, or pandas.DataFrame
#         The axes to compute CSV hash from, or a pre-exported DataFrame
# 
#     Returns
#     -------
#     str or None
#         SHA256 hash of the CSV data (first 16 chars), or None if unable to compute
#     """
#     # If it's already a DataFrame, use the direct hash function
#     if isinstance(ax_or_df, pd.DataFrame):
#         return _compute_csv_hash_from_df(ax_or_df)
# 
#     ax = ax_or_df
# 
#     # Check if we have scitex history with export capability
#     if not hasattr(ax, "export_as_csv"):
#         return None
# 
#     try:
#         # For single axes figures (most common case), ax_index = 0
#         ax_index = 0
# 
#         # Export the data as CSV from the AxisWrapper
#         df = ax.export_as_csv()
# 
#         if df is None or df.empty:
#             return None
# 
#         # Add axis prefix to match what FigWrapper.export_as_csv produces
#         # Uses zero-padded index: ax_00_, ax_01_, etc.
#         prefix = f"ax_{ax_index:02d}_"
#         new_cols = []
#         for col in df.columns:
#             col_str = str(col)
#             if not col_str.startswith(prefix):
#                 col_str = f"{prefix}{col_str}"
#             new_cols.append(col_str)
#         df.columns = new_cols
# 
#         # Convert to CSV string for hashing
#         csv_string = df.to_csv(index=False)
# 
#         # Compute SHA256 hash
#         hash_obj = hashlib.sha256(csv_string.encode("utf-8"))
#         hash_hex = hash_obj.hexdigest()
# 
#         # Return first 16 characters for readability
#         return hash_hex[:16]
# 
#     except Exception:
#         return None

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_csv_hash.py
# --------------------------------------------------------------------------------
