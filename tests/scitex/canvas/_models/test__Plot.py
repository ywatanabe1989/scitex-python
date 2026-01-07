# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_Plot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # File: ./src/scitex/vis/model/plot.py
# """Plot JSON model for scitex.canvas."""
# 
# from dataclasses import asdict, dataclass, field
# from typing import Any, Dict, List, Optional
# 
# from ._Styles import PlotStyle
# 
# 
# @dataclass
# class PlotModel:
#     """
#     Plot model representing a single data visualization within an axes.
# 
#     Separates data/structure from style properties for easier:
#     - UI property panel generation
#     - Style copy/paste
#     - Batch style application
# 
#     Supports common plot types:
#     - line, scatter, bar, barh, hist
#     - errorbar, fill_between
#     - heatmap, imshow, contour, contourf
#     - box, violin
#     """
# 
#     # Plot type
#     plot_type: str  # "line", "scatter", "bar", "errorbar", "heatmap", etc.
# 
#     # Data (can be embedded or referenced)
#     data: Dict[str, Any] = field(default_factory=dict)
# 
#     # Human-readable identifiers
#     plot_id: Optional[str] = None
#     label: Optional[str] = None
#     tags: List[str] = field(default_factory=list)
# 
#     # Style properties (separated for clean UI/copy/paste)
#     style: PlotStyle = field(default_factory=PlotStyle)
# 
#     # Additional kwargs for matplotlib (for advanced use)
#     extra_kwargs: Dict[str, Any] = field(default_factory=dict)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for JSON serialization."""
#         d = asdict(self)
#         d["style"] = self.style.to_dict()
#         return d
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "PlotModel":
#         """Create PlotModel from dictionary."""
#         # Handle old format (backward compatibility)
#         if "style" not in data:
#             # Extract style properties from flat structure
#             style_fields = PlotStyle.__annotations__.keys()
#             style_data = {k: v for k, v in data.items() if k in style_fields}
#             data = {k: v for k, v in data.items() if k not in style_fields}
#             data["style"] = style_data
# 
#         # Extract and parse style
#         style_data = data.pop("style", {})
#         obj = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
#         obj.style = PlotStyle.from_dict(style_data)
#         return obj
# 
#     def validate(self) -> bool:
#         """
#         Validate the plot model.
# 
#         Returns
#         -------
#         bool
#             True if valid, raises ValueError otherwise
#         """
#         # Currently implemented plot types
#         # TODO: Add box, violin, step, stem when renderers are implemented
#         valid_plot_types = [
#             "line",
#             "scatter",
#             "bar",
#             "barh",
#             "hist",
#             "errorbar",
#             "fill_between",
#             "heatmap",
#             "imshow",
#             "contour",
#             "contourf",
#         ]
# 
#         if self.plot_type not in valid_plot_types:
#             raise ValueError(
#                 f"Invalid plot_type: {self.plot_type}. "
#                 f"Must be one of {valid_plot_types}"
#             )
# 
#         # Validate data exists
#         if not self.data:
#             raise ValueError("Plot must have data")
# 
#         # Type-specific validation
#         if self.plot_type in ["heatmap", "imshow"]:
#             if "z" not in self.data and "img" not in self.data:
#                 raise ValueError(f"{self.plot_type} requires 'z' or 'img' in data")
# 
#         elif self.plot_type in ["line", "scatter", "errorbar"]:
#             if "x" not in self.data or "y" not in self.data:
#                 raise ValueError(f"{self.plot_type} requires 'x' and 'y' in data")
# 
#         elif self.plot_type in ["bar", "barh"]:
#             if "x" not in self.data or "height" not in self.data:
#                 # Allow alternative format with y instead of height
#                 if "y" not in self.data:
#                     raise ValueError(
#                         f"{self.plot_type} requires 'x' and 'height' (or 'y') in data"
#                     )
# 
#         return True
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_Plot.py
# --------------------------------------------------------------------------------
