# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_DataSource.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/DataSource.py
# 
# """DataSource - Original data source information."""
# 
# from dataclasses import dataclass
# from typing import Any, Dict, Optional
# 
# 
# @dataclass
# class DataSource:
#     """Original data source information."""
# 
#     path: Optional[str] = None
#     sha256: Optional[str] = None
#     created_at: Optional[str] = None
#     description: Optional[str] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {}
#         if self.path:
#             result["path"] = self.path
#         if self.sha256:
#             result["sha256"] = self.sha256
#         if self.created_at:
#             result["created_at"] = self.created_at
#         if self.description:
#             result["description"] = self.description
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "DataSource":
#         return cls(
#             path=data.get("path"),
#             sha256=data.get("sha256"),
#             created_at=data.get("created_at"),
#             description=data.get("description"),
#         )
# 
# 
# __all__ = ["DataSource"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_DataSource.py
# --------------------------------------------------------------------------------
