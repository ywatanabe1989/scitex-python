# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_ChannelEncoding.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/ChannelEncoding.py
# 
# """ChannelEncoding - Single channel encoding."""
# 
# from dataclasses import dataclass
# from typing import Any, Dict, Optional
# 
# 
# @dataclass
# class ChannelEncoding:
#     """Encoding for a single visual channel (x, y, color, size, etc.)."""
# 
#     column: Optional[str] = None
#     scale: str = "linear"  # linear, log, categorical, ordinal, time
#     domain: Optional[tuple] = None  # Min/max or category list
#     range: Optional[tuple] = None  # Output range for mapping
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {}
#         if self.column:
#             result["column"] = self.column
#         if self.scale != "linear":
#             result["scale"] = self.scale
#         if self.domain:
#             result["domain"] = list(self.domain)
#         if self.range:
#             result["range"] = list(self.range)
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "ChannelEncoding":
#         domain = data.get("domain")
#         range_ = data.get("range")
#         return cls(
#             column=data.get("column"),
#             scale=data.get("scale", "linear"),
#             domain=tuple(domain) if domain else None,
#             range=tuple(range_) if range_ else None,
#         )
# 
# 
# __all__ = ["ChannelEncoding"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_ChannelEncoding.py
# --------------------------------------------------------------------------------
