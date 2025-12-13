#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/model/guides.py
"""Guide elements (lines, spans) JSON model for scitex.fig."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from ._styles import GuideStyle


@dataclass
class GuideModel:
    """
    Guide model for reference lines, spans, and regions.

    Separates structure from style properties for easier:
    - UI property panel generation
    - Style copy/paste
    - Batch style application

    Supports:
    - axhline, axvline: Horizontal/vertical reference lines
    - axhspan, axvspan: Horizontal/vertical shaded regions
    """

    # Guide type
    guide_type: str  # "axhline", "axvline", "axhspan", "axvspan"

    # Position/range (structure, not style)
    y: Optional[float] = None  # For axhline
    x: Optional[float] = None  # For axvline
    ymin: Optional[float] = None  # For axhspan
    ymax: Optional[float] = None  # For axhspan
    xmin: Optional[float] = None  # For axvspan
    xmax: Optional[float] = None  # For axvspan

    # Human-readable identifiers
    label: Optional[str] = None
    guide_id: Optional[str] = None

    # Style properties (separated for clean UI/copy/paste)
    style: GuideStyle = field(default_factory=GuideStyle)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["style"] = self.style.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuideModel":
        """Create GuideModel from dictionary."""
        # Handle old format (backward compatibility)
        if "style" not in data:
            # Extract style properties from flat structure
            style_fields = GuideStyle.__annotations__.keys()
            style_data = {k: v for k, v in data.items() if k in style_fields}
            data = {k: v for k, v in data.items() if k not in style_fields}
            data["style"] = style_data

        # Extract and parse style
        style_data = data.pop("style", {})
        obj = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        obj.style = GuideStyle.from_dict(style_data)
        return obj

    def validate(self) -> bool:
        """
        Validate the guide model.

        Returns
        -------
        bool
            True if valid, raises ValueError otherwise
        """
        valid_guide_types = ["axhline", "axvline", "axhspan", "axvspan"]

        if self.guide_type not in valid_guide_types:
            raise ValueError(
                f"Invalid guide_type: {self.guide_type}. "
                f"Must be one of {valid_guide_types}"
            )

        # Type-specific validation
        if self.guide_type == "axhline":
            if self.y is None:
                raise ValueError("axhline requires 'y' parameter")

        elif self.guide_type == "axvline":
            if self.x is None:
                raise ValueError("axvline requires 'x' parameter")

        elif self.guide_type == "axhspan":
            if self.ymin is None or self.ymax is None:
                raise ValueError("axhspan requires 'ymin' and 'ymax' parameters")

        elif self.guide_type == "axvspan":
            if self.xmin is None or self.xmax is None:
                raise ValueError("axvspan requires 'xmin' and 'xmax' parameters")

        return True


# EOF
