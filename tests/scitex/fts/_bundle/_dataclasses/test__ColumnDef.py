# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_ColumnDef.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/ColumnDef.py
# 
# """ColumnDef - Column definition with metadata."""
# 
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional
# 
# 
# @dataclass
# class ColumnDef:
#     """Column definition with metadata."""
# 
#     name: str
#     dtype: str  # float64, int64, string, bool, datetime, category
#     description: Optional[str] = None
#     unit: Optional[str] = None  # Physical unit: mV, ms, Hz, etc.
#     role: Optional[str] = None  # x, y, group, subject, condition, time, error, weight
#     missing_count: Optional[int] = None
#     unique_count: Optional[int] = None
#     min: Optional[float] = None
#     max: Optional[float] = None
#     mean: Optional[float] = None
#     std: Optional[float] = None
#     categories: Optional[List[str]] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"name": self.name, "dtype": self.dtype}
#         for attr in [
#             "description",
#             "unit",
#             "role",
#             "missing_count",
#             "unique_count",
#             "min",
#             "max",
#             "mean",
#             "std",
#             "categories",
#         ]:
#             val = getattr(self, attr)
#             if val is not None:
#                 result[attr] = val
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Any) -> "ColumnDef":
#         # Handle case where column is just a string (column name)
#         if isinstance(data, str):
#             return cls(name=data, dtype="unknown")
#         if not isinstance(data, dict):
#             return cls(name="", dtype="unknown")  # Default
#         return cls(
#             name=data.get("name", ""),
#             dtype=data.get("dtype", "float64"),
#             description=data.get("description"),
#             unit=data.get("unit"),
#             role=data.get("role"),
#             missing_count=data.get("missing_count"),
#             unique_count=data.get("unique_count"),
#             min=data.get("min"),
#             max=data.get("max"),
#             mean=data.get("mean"),
#             std=data.get("std"),
#             categories=data.get("categories"),
#         )
# 
# 
# __all__ = ["ColumnDef"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_ColumnDef.py
# --------------------------------------------------------------------------------
