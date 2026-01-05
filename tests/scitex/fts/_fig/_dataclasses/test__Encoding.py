# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_Encoding.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/Encoding.py
# 
# """Encoding - Complete encoding specification for a bundle."""
# 
# import json
# from dataclasses import dataclass, field
# from typing import Any, Dict, List
# 
# from ._TraceEncoding import TraceEncoding
# 
# ENCODING_VERSION = "1.0.0"
# 
# 
# @dataclass
# class AxesConfig:
#     """Axis configuration for encoding."""
# 
#     title: str = ""
#     type: str = "quantitative"
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {}
#         if self.title:
#             result["title"] = self.title
#         if self.type != "quantitative":
#             result["type"] = self.type
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "AxesConfig":
#         return cls(
#             title=data.get("title", ""),
#             type=data.get("type", "quantitative"),
#         )
# 
# 
# @dataclass
# class Encoding:
#     """Complete encoding specification for a bundle.
# 
#     Stored in encoding.json at bundle root.
#     """
# 
#     traces: List[TraceEncoding] = field(default_factory=list)
#     axes: Dict[str, AxesConfig] = field(default_factory=dict)  # {"x": AxesConfig, "y": AxesConfig}
# 
#     # Schema metadata
#     schema_name: str = "fsb.encoding"
#     schema_version: str = ENCODING_VERSION
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "traces": [t.to_dict() for t in self.traces],
#         }
#         if self.axes:
#             result["axes"] = {k: v.to_dict() for k, v in self.axes.items()}
#         return result
# 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Encoding":
#         axes = {}
#         axes_data = data.get("axes", {})
#         for key, val in axes_data.items():
#             axes[key] = AxesConfig.from_dict(val) if isinstance(val, dict) else AxesConfig()
#         return cls(
#             traces=[TraceEncoding.from_dict(t) for t in data.get("traces", [])],
#             axes=axes,
#         )
# 
#     @classmethod
#     def from_json(cls, json_str: str) -> "Encoding":
#         return cls.from_dict(json.loads(json_str))
# 
# 
# __all__ = ["ENCODING_VERSION", "AxesConfig", "Encoding"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_Encoding.py
# --------------------------------------------------------------------------------
