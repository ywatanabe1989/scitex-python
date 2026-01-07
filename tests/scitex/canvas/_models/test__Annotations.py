# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_Annotations.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # File: ./src/scitex/vis/model/annotations.py
# """Annotation JSON model for scitex.canvas."""
# 
# from dataclasses import asdict, dataclass, field
# from typing import Any, Dict, Optional, Tuple
# 
# from ._Styles import TextStyle
# 
# 
# @dataclass
# class AnnotationModel:
#     """
#     Annotation model for text, arrows, and shapes on plots.
# 
#     Separates structure from style properties for easier:
#     - UI property panel generation
#     - Style copy/paste
#     - Batch style application
# 
#     Supports:
#     - text: Simple text annotations
#     - annotate: Text with optional arrows
#     - arrow: Arrows without text
#     """
# 
#     # Annotation type
#     annotation_type: str  # "text", "annotate", "arrow"
# 
#     # Text content
#     text: Optional[str] = None
# 
#     # Position (data coordinates)
#     x: float = 0.0
#     y: float = 0.0
# 
#     # For annotate: arrow properties
#     xytext: Optional[Tuple[float, float]] = None  # Text position
# 
#     # Annotation ID for reference
#     annotation_id: Optional[str] = None
# 
#     # Style properties (separated for clean UI/copy/paste)
#     style: TextStyle = field(default_factory=TextStyle)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for JSON serialization."""
#         d = asdict(self)
#         # Convert tuples to lists for JSON compatibility
#         if d.get("xytext") is not None:
#             d["xytext"] = list(d["xytext"])
#         d["style"] = self.style.to_dict()
#         return d
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "AnnotationModel":
#         """Create AnnotationModel from dictionary."""
#         # Convert lists back to tuples
#         if "xytext" in data and data["xytext"] is not None:
#             data["xytext"] = tuple(data["xytext"])
# 
#         # Handle old format (backward compatibility)
#         if "style" not in data:
#             # Extract style properties from flat structure
#             style_fields = TextStyle.__annotations__.keys()
#             style_data = {k: v for k, v in data.items() if k in style_fields}
#             data = {k: v for k, v in data.items() if k not in style_fields}
#             data["style"] = style_data
# 
#         # Extract and parse style
#         style_data = data.pop("style", {})
#         obj = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
#         obj.style = TextStyle.from_dict(style_data)
#         return obj
# 
#     def validate(self) -> bool:
#         """
#         Validate the annotation model.
# 
#         Returns
#         -------
#         bool
#             True if valid, raises ValueError otherwise
#         """
#         valid_annotation_types = ["text", "annotate", "arrow"]
# 
#         if self.annotation_type not in valid_annotation_types:
#             raise ValueError(
#                 f"Invalid annotation_type: {self.annotation_type}. "
#                 f"Must be one of {valid_annotation_types}"
#             )
# 
#         # Type-specific validation
#         if self.annotation_type in ["text", "annotate"]:
#             if not self.text:
#                 raise ValueError(f"{self.annotation_type} requires 'text' parameter")
# 
#         if self.annotation_type == "annotate":
#             if self.xytext is None:
#                 raise ValueError("annotate requires 'xytext' parameter")
# 
#         # Validate alignment (now in style)
#         valid_ha = ["left", "center", "right"]
#         valid_va = ["top", "center", "bottom", "baseline"]
# 
#         if self.style.ha not in valid_ha:
#             raise ValueError(f"Invalid ha: {self.style.ha}. Must be one of {valid_ha}")
# 
#         if self.style.va not in valid_va:
#             raise ValueError(f"Invalid va: {self.style.va}. Must be one of {valid_va}")
# 
#         return True
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_Annotations.py
# --------------------------------------------------------------------------------
