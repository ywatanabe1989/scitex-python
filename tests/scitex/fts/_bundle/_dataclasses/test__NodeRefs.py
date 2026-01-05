# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_NodeRefs.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/NodeRefs.py
# 
# """NodeRefs - References to associated files within the bundle."""
# 
# from dataclasses import dataclass
# from typing import Any, Dict, Optional
# 
# 
# @dataclass
# class NodeRefs:
#     """References to associated files within the bundle.
# 
#     All paths are relative to the bundle root.
#     """
# 
#     encoding: str = "encoding.json"
#     theme: str = "theme.json"
#     data: Optional[str] = None  # data/ directory or specific file
#     stats: str = "stats/stats.json"
# 
#     def to_dict(self) -> Dict[str, str]:
#         result = {
#             "encoding": self.encoding,
#             "theme": self.theme,
#             "stats": self.stats,
#         }
#         if self.data:
#             result["data"] = self.data
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "NodeRefs":
#         return cls(
#             encoding=data.get("encoding", "encoding.json"),
#             theme=data.get("theme", "theme.json"),
#             data=data.get("data"),
#             stats=data.get("stats", "stats/stats.json"),
#         )
# 
# 
# __all__ = ["NodeRefs"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_NodeRefs.py
# --------------------------------------------------------------------------------
