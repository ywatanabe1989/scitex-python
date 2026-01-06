# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_Axes.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # File: ./src/scitex/vis/model/axes.py
# """Axes JSON model for scitex.canvas."""
# 
# from dataclasses import asdict, dataclass, field
# from typing import Any, Dict, List, Optional
# 
# from ._Styles import AxesStyle
# 
# 
# @dataclass
# class AxesModel:
#     """
#     Axes model representing a single subplot.
# 
#     Separates structure/data from style properties for easier:
#     - UI property panel generation
#     - Style copy/paste
#     - Batch style application
# 
#     Contains all configuration for one subplot including:
#     - Plot data and type
#     - Axis labels, limits, scales
#     - Legends, titles
#     - Annotations and guides
#     """
# 
#     # Position in layout (0-based)
#     row: int = 0
#     col: int = 0
# 
#     # Plot configurations (list of PlotModel)
#     plots: List[Dict[str, Any]] = field(default_factory=list)
# 
#     # Axis labels and titles (content, not style)
#     xlabel: Optional[str] = None
#     ylabel: Optional[str] = None
#     title: Optional[str] = None
# 
#     # Axis limits
#     xlim: Optional[List[float]] = None
#     ylim: Optional[List[float]] = None
# 
#     # Axis scales
#     xscale: str = "linear"  # "linear", "log", "symlog", "logit"
#     yscale: str = "linear"
# 
#     # Ticks
#     xticks: Optional[List[float]] = None
#     yticks: Optional[List[float]] = None
#     xticklabels: Optional[List[str]] = None
#     yticklabels: Optional[List[str]] = None
# 
#     # Annotations (list of AnnotationModel)
#     annotations: List[Dict[str, Any]] = field(default_factory=list)
# 
#     # Guides (lines, spans, etc.)
#     guides: List[Dict[str, Any]] = field(default_factory=list)
# 
#     # Axes ID for reference
#     axes_id: Optional[str] = None
# 
#     # Style properties (separated for clean UI/copy/paste)
#     style: AxesStyle = field(default_factory=AxesStyle)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for JSON serialization."""
#         d = asdict(self)
#         d["style"] = self.style.to_dict()
#         return d
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "AxesModel":
#         """Create AxesModel from dictionary."""
#         # Handle old format (backward compatibility)
#         if "style" not in data:
#             # Extract style properties from flat structure
#             style_fields = AxesStyle.__annotations__.keys()
#             style_data = {k: v for k, v in data.items() if k in style_fields}
#             data = {k: v for k, v in data.items() if k not in style_fields}
#             data["style"] = style_data
# 
#         # Extract and parse style
#         style_data = data.pop("style", {})
#         obj = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
#         obj.style = AxesStyle.from_dict(style_data)
#         return obj
# 
#     def add_plot(self, plot_config: Dict[str, Any]) -> None:
#         """
#         Add a plot configuration to this axes.
# 
#         Parameters
#         ----------
#         plot_config : Dict[str, Any]
#             Plot configuration dictionary
#         """
#         self.plots.append(plot_config)
# 
#     def add_annotation(self, annotation_config: Dict[str, Any]) -> None:
#         """
#         Add an annotation to this axes.
# 
#         Parameters
#         ----------
#         annotation_config : Dict[str, Any]
#             Annotation configuration dictionary
#         """
#         self.annotations.append(annotation_config)
# 
#     def add_guide(self, guide_config: Dict[str, Any]) -> None:
#         """
#         Add a guide (line, span, etc.) to this axes.
# 
#         Parameters
#         ----------
#         guide_config : Dict[str, Any]
#             Guide configuration dictionary
#         """
#         self.guides.append(guide_config)
# 
#     def validate(self) -> bool:
#         """
#         Validate the axes model.
# 
#         Returns
#         -------
#         bool
#             True if valid, raises ValueError otherwise
#         """
#         if self.row < 0:
#             raise ValueError(f"row must be non-negative, got {self.row}")
# 
#         if self.col < 0:
#             raise ValueError(f"col must be non-negative, got {self.col}")
# 
#         if self.xscale not in ["linear", "log", "symlog", "logit"]:
#             raise ValueError(f"Invalid xscale: {self.xscale}")
# 
#         if self.yscale not in ["linear", "log", "symlog", "logit"]:
#             raise ValueError(f"Invalid yscale: {self.yscale}")
# 
#         if self.xlim is not None and len(self.xlim) != 2:
#             raise ValueError(f"xlim must have 2 elements, got {len(self.xlim)}")
# 
#         if self.ylim is not None and len(self.ylim) != 2:
#             raise ValueError(f"ylim must have 2 elements, got {len(self.ylim)}")
# 
#         return True
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_Axes.py
# --------------------------------------------------------------------------------
