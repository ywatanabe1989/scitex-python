# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_Theme.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/Theme.py
# 
# """Theme - Visual aesthetics for bundles."""
# 
# import json
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional
# 
# THEME_VERSION = "1.0.0"
# 
# 
# @dataclass
# class Colors:
#     """Color palette configuration."""
# 
#     palette: List[str] = field(
#         default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c"]
#     )
#     background: str = "#ffffff"
#     text: str = "#000000"
#     axis: str = "#333333"
#     grid: str = "#e0e0e0"
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "palette": self.palette,
#             "background": self.background,
#             "text": self.text,
#             "axis": self.axis,
#             "grid": self.grid,
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Colors":
#         return cls(
#             palette=data.get("palette", ["#1f77b4", "#ff7f0e", "#2ca02c"]),
#             background=data.get("background", "#ffffff"),
#             text=data.get("text", "#000000"),
#             axis=data.get("axis", "#333333"),
#             grid=data.get("grid", "#e0e0e0"),
#         )
# 
# 
# @dataclass
# class Typography:
#     """Typography configuration."""
# 
#     family: str = "sans-serif"
#     size_pt: float = 8.0
#     title_size_pt: float = 10.0
#     label_size_pt: float = 8.0
#     tick_size_pt: float = 7.0
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "family": self.family,
#             "size_pt": self.size_pt,
#             "title_size_pt": self.title_size_pt,
#             "label_size_pt": self.label_size_pt,
#             "tick_size_pt": self.tick_size_pt,
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Typography":
#         return cls(
#             family=data.get("family", "sans-serif"),
#             size_pt=data.get("size_pt", 8.0),
#             title_size_pt=data.get("title_size_pt", 10.0),
#             label_size_pt=data.get("label_size_pt", 8.0),
#             tick_size_pt=data.get("tick_size_pt", 7.0),
#         )
# 
# 
# @dataclass
# class Lines:
#     """Line style configuration."""
# 
#     width_pt: float = 1.0
#     style: str = "solid"  # solid, dashed, dotted, dashdot
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {"width_pt": self.width_pt, "style": self.style}
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Lines":
#         return cls(
#             width_pt=data.get("width_pt", 1.0),
#             style=data.get("style", "solid"),
#         )
# 
# 
# @dataclass
# class Markers:
#     """Marker style configuration."""
# 
#     size_pt: float = 4.0
#     symbol: str = "circle"  # circle, square, triangle, cross, diamond
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {"size_pt": self.size_pt, "symbol": self.symbol}
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Markers":
#         return cls(
#             size_pt=data.get("size_pt", 4.0),
#             symbol=data.get("symbol", "circle"),
#         )
# 
# 
# @dataclass
# class Grid:
#     """Grid line configuration."""
# 
#     show: bool = True
#     major_width_pt: float = 0.5
#     minor_width_pt: float = 0.25
#     major_alpha: float = 0.3
#     minor_alpha: float = 0.1
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "show": self.show,
#             "major_width_pt": self.major_width_pt,
#             "minor_width_pt": self.minor_width_pt,
#             "major_alpha": self.major_alpha,
#             "minor_alpha": self.minor_alpha,
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Any) -> "Grid":
#         # Handle case where grid is a boolean (e.g., grid: true)
#         if isinstance(data, bool):
#             return cls(show=data)
#         if not isinstance(data, dict):
#             return cls()  # Default
#         return cls(
#             show=data.get("show", True),
#             major_width_pt=data.get("major_width_pt", 0.5),
#             minor_width_pt=data.get("minor_width_pt", 0.25),
#             major_alpha=data.get("major_alpha", 0.3),
#             minor_alpha=data.get("minor_alpha", 0.1),
#         )
# 
# 
# @dataclass
# class TraceTheme:
#     """Per-trace theme overrides."""
# 
#     trace_id: str
#     color: Optional[str] = None
#     line_width_pt: Optional[float] = None
#     marker_size_pt: Optional[float] = None
#     alpha: Optional[float] = None
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"trace_id": self.trace_id}
#         if self.color:
#             result["color"] = self.color
#         if self.line_width_pt is not None:
#             result["line_width_pt"] = self.line_width_pt
#         if self.marker_size_pt is not None:
#             result["marker_size_pt"] = self.marker_size_pt
#         if self.alpha is not None:
#             result["alpha"] = self.alpha
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "TraceTheme":
#         return cls(
#             trace_id=data.get("trace_id", "trace_0"),
#             color=data.get("color"),
#             line_width_pt=data.get("line_width_pt"),
#             marker_size_pt=data.get("marker_size_pt"),
#             alpha=data.get("alpha"),
#         )
# 
# 
# @dataclass
# class Theme:
#     """Complete theme specification for a bundle.
# 
#     Stored in theme.json at bundle root.
#     """
# 
#     colors: Colors = field(default_factory=Colors)
#     typography: Typography = field(default_factory=Typography)
#     lines: Lines = field(default_factory=Lines)
#     markers: Markers = field(default_factory=Markers)
#     grid: Grid = field(default_factory=Grid)
#     traces: List[TraceTheme] = field(default_factory=list)
#     preset: Optional[str] = None  # nature, science, dark, minimal
# 
#     # Schema metadata
#     schema_name: str = "fsb.theme"
#     schema_version: str = THEME_VERSION
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "colors": self.colors.to_dict(),
#             "typography": self.typography.to_dict(),
#             "lines": self.lines.to_dict(),
#             "markers": self.markers.to_dict(),
#             "grid": self.grid.to_dict(),
#         }
#         if self.traces:
#             result["traces"] = [t.to_dict() for t in self.traces]
#         if self.preset:
#             result["preset"] = self.preset
#         return result
# 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Theme":
#         return cls(
#             colors=Colors.from_dict(data.get("colors", {})),
#             typography=Typography.from_dict(data.get("typography", {})),
#             lines=Lines.from_dict(data.get("lines", {})),
#             markers=Markers.from_dict(data.get("markers", {})),
#             grid=Grid.from_dict(data.get("grid", {})),
#             traces=[TraceTheme.from_dict(t) for t in data.get("traces", [])],
#             preset=data.get("preset"),
#         )
# 
#     @classmethod
#     def from_json(cls, json_str: str) -> "Theme":
#         return cls.from_dict(json.loads(json_str))
# 
# 
# __all__ = [
#     "THEME_VERSION",
#     "Colors",
#     "Typography",
#     "Lines",
#     "Markers",
#     "Grid",
#     "TraceTheme",
#     "Theme",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_dataclasses/_Theme.py
# --------------------------------------------------------------------------------
