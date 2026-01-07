# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_TraceEncoding.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/TraceEncoding.py
# 
# """TraceEncoding - Encoding for a single trace/data series."""
# 
# from dataclasses import dataclass
# from typing import Any, Dict, Optional
# 
# from ._ChannelEncoding import ChannelEncoding
# 
# 
# @dataclass
# class TraceEncoding:
#     """Encoding specification for a single trace/data series."""
# 
#     trace_id: str
#     data_ref: Optional[str] = None  # Path to data file within bundle
#     x: Optional[ChannelEncoding] = None
#     y: Optional[ChannelEncoding] = None
#     color: Optional[ChannelEncoding] = None
#     size: Optional[ChannelEncoding] = None
#     group: Optional[ChannelEncoding] = None
#     label: Optional[ChannelEncoding] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"trace_id": self.trace_id}
#         if self.data_ref:
#             result["data_ref"] = self.data_ref
#         for channel in ["x", "y", "color", "size", "group", "label"]:
#             enc = getattr(self, channel)
#             if enc:
#                 result[channel] = enc.to_dict()
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "TraceEncoding":
#         return cls(
#             trace_id=data.get("trace_id", "trace_0"),
#             data_ref=data.get("data_ref"),
#             x=ChannelEncoding.from_dict(data["x"]) if "x" in data else None,
#             y=ChannelEncoding.from_dict(data["y"]) if "y" in data else None,
#             color=ChannelEncoding.from_dict(data["color"]) if "color" in data else None,
#             size=ChannelEncoding.from_dict(data["size"]) if "size" in data else None,
#             group=ChannelEncoding.from_dict(data["group"]) if "group" in data else None,
#             label=ChannelEncoding.from_dict(data["label"]) if "label" in data else None,
#         )
# 
# 
# __all__ = ["TraceEncoding"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_TraceEncoding.py
# --------------------------------------------------------------------------------
