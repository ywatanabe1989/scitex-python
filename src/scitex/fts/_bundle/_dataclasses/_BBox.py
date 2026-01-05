#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/BBox.py

"""BBox - Bounding box in normalized coordinates (0-1)."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BBox:
    """Bounding box in normalized coordinates (0-1).

    Used for positioning elements within parent containers.
    Origin is top-left, y increases downward.
    """

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0

    def __post_init__(self):
        # Clamp to valid range
        self.x0 = max(0.0, min(1.0, self.x0))
        self.y0 = max(0.0, min(1.0, self.y0))
        self.x1 = max(0.0, min(1.0, self.x1))
        self.y1 = max(0.0, min(1.0, self.y1))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BBox":
        return cls(
            x0=data.get("x0", 0.0),
            y0=data.get("y0", 0.0),
            x1=data.get("x1", 1.0),
            y1=data.get("y1", 1.0),
        )


__all__ = ["BBox"]

# EOF
