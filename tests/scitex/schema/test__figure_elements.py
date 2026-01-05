# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_figure_elements.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-19
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_figure_elements.py
# 
# """
# Figure Elements Schema - Title, Caption, and Panel Labels.
# 
# This module defines figure-level text elements that appear in theme.json
# for editability while maintaining consistent styling across the figure.
# 
# These elements enable auto-generation of publication-ready captions:
#   "Figure 1: Main Results. (A) Time-series analysis. (B) Frequency distribution."
# """
# 
# from dataclasses import asdict, dataclass, field
# from typing import Any, Dict, List, Optional
# 
# FIGURE_ELEMENTS_VERSION = "1.0.0"
# 
# 
# @dataclass
# class FigureTitle:
#     """
#     Figure title specification.
# 
#     Parameters
#     ----------
#     text : str
#         The main title text (e.g., "Main Results")
#     prefix : str
#         Prefix before number (e.g., "Figure", "Fig.")
#     number : int, optional
#         Figure number (None for unnumbered)
#     font_size_pt : float
#         Title font size in points
#     font_weight : str
#         Font weight (normal, bold)
#     position : str
#         Position (top, bottom)
#     visible : bool
#         Whether to render the title
#     """
# 
#     text: str = ""
#     prefix: str = "Figure"
#     number: Optional[int] = None
#     font_size_pt: float = 10.0
#     font_weight: str = "bold"
#     position: str = "top"
#     visible: bool = True
# 
#     def format(self, include_number: bool = True) -> str:
#         """Format the title as displayed string."""
#         if include_number and self.number is not None:
#             return f"{self.prefix} {self.number}: {self.text}"
#         elif self.text:
#             return self.text
#         return ""
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {
#             "text": self.text,
#             "prefix": self.prefix,
#             "font_size_pt": self.font_size_pt,
#             "font_weight": self.font_weight,
#             "position": self.position,
#             "visible": self.visible,
#         }
#         if self.number is not None:
#             result["number"] = self.number
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "FigureTitle":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class Caption:
#     """
#     Figure caption specification.
# 
#     Parameters
#     ----------
#     text : str
#         Manual caption text (overrides auto-generation if set)
#     auto_generate : bool
#         Whether to auto-generate from panel descriptions
#     font_size_pt : float
#         Caption font size in points
#     position : str
#         Position (top, bottom)
#     visible : bool
#         Whether to render the caption
#     """
# 
#     text: str = ""
#     auto_generate: bool = True
#     font_size_pt: float = 8.0
#     position: str = "bottom"
#     visible: bool = True
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return asdict(self)
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Caption":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# @dataclass
# class PanelLabels:
#     """
#     Panel label styling for multi-panel figures.
# 
#     Controls how panel letters (A, B, C, ...) are rendered on child plots.
# 
#     Parameters
#     ----------
#     style : str
#         Letter style: "uppercase" (A, B, C), "lowercase" (a, b, c),
#         "roman" (i, ii, iii), "Roman" (I, II, III)
#     format : str
#         Format template with {letter} placeholder
#     font_size_pt : float
#         Label font size in points
#     font_weight : str
#         Font weight (normal, bold)
#     position : str
#         Position on panel (top-left, top-right, bottom-left, bottom-right)
#     offset_mm : dict
#         Offset from corner {x: mm, y: mm}
#     visible : bool
#         Whether to render panel labels
#     """
# 
#     style: str = "uppercase"
#     format: str = "({letter})"
#     font_size_pt: float = 12.0
#     font_weight: str = "bold"
#     position: str = "top-left"
#     offset_mm: Dict[str, float] = field(default_factory=lambda: {"x": 2.0, "y": 2.0})
#     visible: bool = True
# 
#     def format_letter(self, index: int) -> str:
#         """
#         Format panel letter for given index (0-based).
# 
#         Parameters
#         ----------
#         index : int
#             Zero-based panel index
# 
#         Returns
#         -------
#         str
#             Formatted label like "(A)", "(B)", etc.
#         """
#         if self.style == "uppercase":
#             letter = chr(ord("A") + index)
#         elif self.style == "lowercase":
#             letter = chr(ord("a") + index)
#         elif self.style == "roman":
#             letter = _to_roman(index + 1).lower()
#         elif self.style == "Roman":
#             letter = _to_roman(index + 1)
#         else:
#             letter = chr(ord("A") + index)
# 
#         return self.format.replace("{letter}", letter)
# 
#     def get_letter(self, index: int) -> str:
#         """Get just the letter without formatting."""
#         if self.style == "uppercase":
#             return chr(ord("A") + index)
#         elif self.style == "lowercase":
#             return chr(ord("a") + index)
#         elif self.style == "roman":
#             return _to_roman(index + 1).lower()
#         elif self.style == "Roman":
#             return _to_roman(index + 1)
#         return chr(ord("A") + index)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "style": self.style,
#             "format": self.format,
#             "font_size_pt": self.font_size_pt,
#             "font_weight": self.font_weight,
#             "position": self.position,
#             "offset_mm": self.offset_mm,
#             "visible": self.visible,
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "PanelLabels":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# def _to_roman(num: int) -> str:
#     """Convert integer to Roman numeral."""
#     values = [
#         (1000, "M"),
#         (900, "CM"),
#         (500, "D"),
#         (400, "CD"),
#         (100, "C"),
#         (90, "XC"),
#         (50, "L"),
#         (40, "XL"),
#         (10, "X"),
#         (9, "IX"),
#         (5, "V"),
#         (4, "IV"),
#         (1, "I"),
#     ]
#     result = ""
#     for value, numeral in values:
#         while num >= value:
#             result += numeral
#             num -= value
#     return result
# 
# 
# @dataclass
# class PanelInfo:
#     """
#     Information about a single panel in a multi-panel figure.
# 
#     Used in spec.json to store per-panel metadata that combines with
#     theme-level PanelLabels to generate captions.
# 
#     Parameters
#     ----------
#     panel_id : str
#         Unique identifier for the panel (matches child ID)
#     letter : str
#         Panel letter override (auto-assigned if empty)
#     description : str
#         Description for caption generation
#     order : int
#         Display order (0-based)
#     """
# 
#     panel_id: str
#     letter: str = ""
#     description: str = ""
#     order: int = 0
# 
#     def to_dict(self) -> Dict[str, Any]:
#         result = {"panel_id": self.panel_id, "order": self.order}
#         if self.letter:
#             result["letter"] = self.letter
#         if self.description:
#             result["description"] = self.description
#         return result
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "PanelInfo":
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
# 
# 
# def generate_caption(
#     title: FigureTitle,
#     caption: Caption,
#     panels: List[PanelInfo],
#     panel_labels: PanelLabels,
# ) -> str:
#     """
#     Generate full figure caption from components.
# 
#     Format: "Figure 1: Main Title. (A) Description A. (B) Description B."
# 
#     Parameters
#     ----------
#     title : FigureTitle
#         Figure title specification
#     caption : Caption
#         Caption specification
#     panels : list of PanelInfo
#         Panel information with descriptions
#     panel_labels : PanelLabels
#         Panel label styling
# 
#     Returns
#     -------
#     str
#         Complete formatted caption
#     """
#     # If manual caption is set and auto_generate is off, use it
#     if caption.text and not caption.auto_generate:
#         return caption.text
# 
#     parts = []
# 
#     # Add title
#     title_str = title.format()
#     if title_str:
#         parts.append(title_str)
# 
#     # Sort panels by order
#     sorted_panels = sorted(panels, key=lambda p: p.order)
# 
#     # Add panel descriptions
#     panel_parts = []
#     for idx, panel in enumerate(sorted_panels):
#         if panel.description:
#             letter = panel.letter or panel_labels.get_letter(idx)
#             formatted = panel_labels.format.replace("{letter}", letter)
#             panel_parts.append(f"{formatted} {panel.description}")
# 
#     if panel_parts:
#         parts.append(" ".join(panel_parts))
# 
#     return " ".join(parts) if parts else ""
# 
# 
# def generate_caption_latex(
#     title: FigureTitle,
#     caption: Caption,
#     panels: List[PanelInfo],
#     panel_labels: PanelLabels,
# ) -> str:
#     """
#     Generate LaTeX-formatted figure caption.
# 
#     Format: "\\textbf{Figure 1: Main Title.} (A) Description A. (B) Description B."
# 
#     Returns
#     -------
#     str
#         LaTeX-formatted caption
#     """
#     parts = []
# 
#     # Add bold title
#     title_str = title.format()
#     if title_str:
#         parts.append(f"\\textbf{{{title_str}}}")
# 
#     # Sort and add panel descriptions
#     sorted_panels = sorted(panels, key=lambda p: p.order)
#     panel_parts = []
#     for idx, panel in enumerate(sorted_panels):
#         if panel.description:
#             letter = panel.letter or panel_labels.get_letter(idx)
#             formatted = panel_labels.format.replace("{letter}", letter)
#             panel_parts.append(f"\\textbf{{{formatted}}} {panel.description}")
# 
#     if panel_parts:
#         parts.append(" ".join(panel_parts))
# 
#     return " ".join(parts) if parts else ""
# 
# 
# def generate_caption_markdown(
#     title: FigureTitle,
#     caption: Caption,
#     panels: List[PanelInfo],
#     panel_labels: PanelLabels,
# ) -> str:
#     """
#     Generate Markdown-formatted figure caption.
# 
#     Format: "**Figure 1: Main Title.** (A) Description A. (B) Description B."
# 
#     Returns
#     -------
#     str
#         Markdown-formatted caption
#     """
#     parts = []
# 
#     # Add bold title
#     title_str = title.format()
#     if title_str:
#         parts.append(f"**{title_str}**")
# 
#     # Sort and add panel descriptions
#     sorted_panels = sorted(panels, key=lambda p: p.order)
#     panel_parts = []
#     for idx, panel in enumerate(sorted_panels):
#         if panel.description:
#             letter = panel.letter or panel_labels.get_letter(idx)
#             formatted = panel_labels.format.replace("{letter}", letter)
#             panel_parts.append(f"**{formatted}** {panel.description}")
# 
#     if panel_parts:
#         parts.append(" ".join(panel_parts))
# 
#     return " ".join(parts) if parts else ""
# 
# 
# __all__ = [
#     "FIGURE_ELEMENTS_VERSION",
#     "FigureTitle",
#     "Caption",
#     "PanelLabels",
#     "PanelInfo",
#     "generate_caption",
#     "generate_caption_latex",
#     "generate_caption_markdown",
# ]
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/schema/_figure_elements.py
# --------------------------------------------------------------------------------
